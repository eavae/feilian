import bs4
from feilian.soup_tools import prune_by_structure


if __name__ == "__main__":
    struct_text = open(
        "tests/soup_tools/prune_by_structure/case_0.struct.html", "r"
    ).read()
    input_text = open(
        "tests/soup_tools/prune_by_structure/case_0.input.html", "r"
    ).read()

    structure = bs4.BeautifulSoup(struct_text, "html5lib")
    input_soup = bs4.BeautifulSoup(input_text, "html5lib")

    prune_by_structure(input_soup, structure)
    pass
