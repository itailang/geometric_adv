# import system modules
import os.path as osp
import sys

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from transfer.atlasnet.auxiliary.netvision.HtmlGenerator import HtmlGenerator


def main():
    """
    Create a master webpage to summarize results of all experiments.
    Author: Thibault Groueix 01.11.2019
    """
    webpage = HtmlGenerator(path="master.html")

    for dataset in ["Shapenet"]:
        table = webpage.add_table(dataset)
        table.add_column("Num Primitives")
        table.add_column("Decoder")
        table.add_column("Chamfer")
        table.add_column("F-Score")
        table.add_column("Metro")
        table.add_column("Dirname")

    webpage.return_html(save_editable_version=True)


if __name__ == "__main__":
    main()
