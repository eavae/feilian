from feilian.soup_tools import get_structure

if __name__ == "__main__":
    html_content = open("tests/soup_tools/get_structure/case_0.input.html", "r").read()

    structure = get_structure(html_content)
    pass
