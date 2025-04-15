import configparser
import dataclasses
import json
import re
import os
from typing import List

import retry
import google.generativeai as genai
from tqdm import tqdm

from arxiv_scraper import Paper
from arxiv_scraper import EnhancedJSONEncoder


def filter_by_author(all_authors, papers, author_targets, config):
    # filter and parse the papers
    selected_papers = {}  # pass to output
    all_papers = {}  # dict for later filtering
    sort_dict = {}  # dict storing key and score

    # author based selection
    for paper in papers:
        all_papers[paper.arxiv_id] = paper
        for author in paper.authors:
            if author in all_authors:
                for alias in all_authors[author]:
                    if alias["authorId"] in author_targets:
                        selected_papers[paper.arxiv_id] = {
                            **dataclasses.asdict(paper),
                            **{"COMMENT": "Author match"},
                        }
                        sort_dict[paper.arxiv_id] = float(
                            config["SELECTION"]["author_match_score"]
                        )
                        break
    return selected_papers, all_papers, sort_dict


def filter_papers_by_hindex(all_authors, papers, config):
    # filters papers by checking to see if there's at least one author with > hcutoff hindex
    paper_list = []
    for paper in papers:
        max_h = 0
        for author in paper.authors:
            if author in all_authors:
                max_h = max(
                    max_h, max([alias["hIndex"] for alias in all_authors[author]])
                )
        if max_h >= float(config["FILTERING"]["hcutoff"]):
            paper_list.append(paper)
    return paper_list


def calc_price(model, usage):
    # Gemini API는 현재 무료이므로 0을 반환
    return 0.0


@retry.retry(tries=3, delay=2)
def call_gemini(full_prompt, model):
    try:
        # 전달받은 모델 객체를 사용
        if model is None:
            raise ValueError("Gemini model instance is not provided")
        response = model.generate_content(full_prompt)
        return response
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {str(e)}")
        raise


def run_and_parse_gemini(full_prompt, model, config):
    # Gemini API를 호출하고 결과를 파싱
    response = call_gemini(full_prompt, model)
    out_text = response.text
    out_text = re.sub("```jsonl\n", "", out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()
    # split out_text line by line and parse each as a json.
    json_dicts = []
    for line in out_text.split("\n"):
        # try catch block to attempt to parse json
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                print(response.text)
            continue
    return json_dicts, 0.0  # Gemini API는 현재 무료이므로 비용은 0


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    new_str = (
        "ArXiv ID: "
        + paper_entry.arxiv_id
        + "\n"
        + "Title: "
        + paper_entry.title
        + "\n"
        + "Authors: "
        + " and ".join(paper_entry.authors)
        + "\n"
        + "Abstract: "
        + paper_entry.abstract[:4000]
    )
    return new_str


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def filter_papers_by_title(
    papers, config, model, base_prompt, criterion
) -> List[Paper]:
    filter_postfix = 'Identify any papers that are absolutely and completely irrelavent to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of arxiv ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. Even if every paper seems irrelevant, please keep at least TWO papers'
    batches_of_papers = batched(papers, 20)
    final_list = []
    cost = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        response = call_gemini(full_prompt, model)
        out_text = response.text
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id not in filtered_set:
                    final_list.append(paper)
                else:
                    print("Filtered out paper " + paper.arxiv_id)
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(response)
            continue
    return final_list, cost


def paper_to_titles(paper_entry: Paper) -> str:
    return "ArXiv ID: " + paper_entry.arxiv_id + " Title: " + paper_entry.title + "\n"


def run_on_batch(
    paper_batch, base_prompt, criterion, postfix_prompt, model, config
):
    batch_str = [paper_to_string(paper) for paper in paper_batch]
    full_prompt = "\n".join(
        [
            base_prompt,
            criterion + "\n",
            "\n\n".join(batch_str) + "\n",
            postfix_prompt,
        ]
    )
    json_dicts, cost = run_and_parse_gemini(full_prompt, model, config)
    return json_dicts, cost


def filter_by_gpt(
    all_authors, papers, config, model, all_papers, selected_papers, sort_dict
):
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    all_cost = 0
    if config["SELECTION"].getboolean("run_openai"):
        # filter first by hindex of authors to reduce costs.
        paper_list = filter_papers_by_hindex(all_authors, papers, config)
        if config["OUTPUT"].getboolean("debug_messages"):
            print(str(len(paper_list)) + " papers after hindex filtering")
        cost = 0
        paper_list, cost = filter_papers_by_title(
            paper_list, config, model, base_prompt, criterion
        )
        if config["OUTPUT"].getboolean("debug_messages"):
            print(
                str(len(paper_list))
                + " papers after title filtering with cost of $"
                + str(cost)
            )
        all_cost += cost

        # batch the remaining papers and invoke Gemini
        batch_of_papers = batched(paper_list, int(config["SELECTION"]["batch_size"]))
        scored_batches = []
        for batch in tqdm(batch_of_papers):
            scored_in_batch = []
            json_dicts, cost = run_on_batch(
                batch, base_prompt, criterion, postfix_prompt, model, config
            )
            all_cost += cost
            for jdict in json_dicts:
                if (
                    int(jdict["RELEVANCE"])
                    >= int(config["FILTERING"]["relevance_cutoff"])
                    and jdict["NOVELTY"] >= int(config["FILTERING"]["novelty_cutoff"])
                    and jdict["ARXIVID"] in all_papers
                ):
                    selected_papers[jdict["ARXIVID"]] = {
                        **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                        **jdict,
                    }
                    sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]
                scored_in_batch.append(
                    {
                        **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                        **jdict,
                    }
                )
            scored_batches.append(scored_in_batch)
        if config["OUTPUT"].getboolean("dump_debug_file"):
            with open(
                config["OUTPUT"]["output_path"] + "gpt_paper_batches.debug.json", "w"
            ) as outfile:
                json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)
        if config["OUTPUT"].getboolean("debug_messages"):
            print("Total cost: $" + str(all_cost))


if __name__ == "__main__":
    # 환경 변수에서 GEMINI_KEY를 가져옵니다.
    gemini_key = os.environ.get("GEMINI_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_KEY environment variable is not set")

    # Gemini 모델 객체를 생성합니다.
    model = genai.GenerativeModel('gemini-pro', api_key=gemini_key)

    # 설정 파일을 로드합니다.
    config = configparser.ConfigParser()
    config.read('config.ini')

    # 필요한 데이터 구조를 초기화합니다.
    all_authors = {}  # 저자 정보를 저장할 딕셔너리
    papers = []  # 논문 목록
    all_papers = {}  # 모든 논문 정보
    selected_papers = {}  # 선택된 논문 정보
    sort_dict = {}  # 정렬을 위한 딕셔너리

    # filter_by_gpt 함수를 호출합니다.
    filter_by_gpt(all_authors, papers, config, model, all_papers, selected_papers, sort_dict)
