#! /usr/bin/env python3


def options_dict(options):
    opt_dict = {"n": range(2, 11), "isovalue": 0.3, "size": 1.0, "method": "dbscan", "onlypos": True, "doint": False, "seed": 0, "verbose": True}

    for i, op in enumerate(options[0::2]):
        if op == "--help":
            print(
                "To run NCICLUSTER do: ./ncicluster.py input_names [OPTIONS]",
                "Options:",
                "  -n N               set the number of clusters to the int value N",
                "  --isovalue i       set the isovalue to i",
                "  --size s           set the size of the sample to s",
		"  --method m	      choose the clustering method m=\"kmeans\" or m=\"dbscan\"",
                "  --onlypos b        choose if only position is considered (b=True) or not (b=False)",
                "  --doint b          choose if integrals over clustering regions should be computed (b=True) or not (b=False)",
                "  --seed sd          choose seed for clustering, default is 0",
                "  -v V               choose verbose mode, default is True",
                "  --help             display this help and exit",
                sep="\n",
            )
            exit()
        else:
            if op == "-n":
                opt_dict["n"] = int(options[2 * i + 1])
            elif op == "--isovalue":
                opt_dict["isovalue"] = float(options[2 * i + 1])
            elif op == "--size":
                opt_dict["size"] = float(options[2 * i + 1])
            elif op == "--method":
                opt_dict["method"] = options[2 * i + 1]
            elif op == "--seed":
                opt_dict["seed"] = options[2 * i + 1]
            elif op == "-v":
                if options[2 * i + 1] == "True":
                    opt_dict["verbose"] = True
                elif options[2 * i + 1] == "False":
                    opt_dict["verbose"] = False
                else:
                    raise ValueError(
                        "{} is not a valid option for -v. Try True or False,".format(
                            options[2 * i + 1]
                        )
                    )
            elif op == "--onlypos":
                if options[2 * i + 1] == "True":
                    opt_dict["onlypos"] = True
                elif options[2 * i + 1] == "False":
                    opt_dict["onlypos"] = False
                else:
                    raise ValueError(
                        "{} is not a valid option for --onlypos. Try True or False,".format(
                            options[2 * i + 1]
                        )
                    )
            elif op == "--doint":
                if options[2 * i + 1] == "True":
                    opt_dict["doint"] = True
                elif options[2 * i + 1] == "False":
                    opt_dict["doint"] = False
                else:
                    raise ValueError(
                        "{} is not a valid option for --doint. Try True or False,".format(
                            options[2 * i + 1]
                        )
                    )
            else:
                raise ValueError("{} is not a valid option".format(op))

    return opt_dict
