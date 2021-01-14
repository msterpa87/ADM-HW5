import random
from collections import defaultdict


def categories_preprocess():
    """ Break ties u.a.r. for articles appearing in multiple categories
        Saves the result in the file reduced_categories.txt """

    pages_dict = defaultdict(list)

    # read file content into a dictionary {Page: [category1, category2, ...]}
    with open("wiki-topcats-categories.txt", "r") as f:
        for line in f.readlines():
            category, pages = line.strip().split(';')
            category = category.split(':')[1]
            pages = list(map(int, pages.split()))

            for page in pages:
                pages_dict[page].append(category)

    single_category_dict = defaultdict(list)

    # break ties u.a.r. and creates dict {Category: [page1, page2, ...]}
    for page, category_list in pages_dict.items():
        if len(category_list) == 1:
            single_category_dict[category_list[0]].append(page)
            continue

        # multiple categories, breaking tie u.a.r.
        random_category = random.choice(category_list)
        single_category_dict[random_category].append(page)

    # saves dictionary to file
    with open("reduced_categories.txt", "w") as f:
        for category, pages in single_category_dict.items():
            line = "{}: {}\n".format(category, " ".join(list(map(str, pages))))
            f.write(line)
