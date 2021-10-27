# NCICLUSTER 

Developed by Trinidad Novoa, Francesca Peccati and Julia
Contreras-García at Laboratoire de Chimie Théorique,
Sorbonne Université.

Contact email: trinidad.novoa_aguirre@upmc.fr


NCICLUSTER is a part of the NCIPLOT program to identify
non-covalent interaction regions. NCICLUSTER takes the
output cube file of NCIPLOT and separates the interaction
regions according to their spatial position. NCICLUSTER 
has been written completely in Python3. 


To launch NCICLUSTER, do

    python3 ncicluster.py input_names --opt opt_val

where input_names is a file with the common path to the
NCIPLOT output cube files, one per line. This is, if the
user would like to analize system1 with cube files
/path1/system1-dens.cube and /path1/system1-grad.cube, 
the input_names file should have one line: /path1/system1.
The output files are to be saved in the same directory.

The user can run 

    python3 ncicluster.py --help

to display the options menu, which goes as follows.

Options:
  -n N               set the number of clusters to the int value N
  --isovalue i       set the isovalue to i
  --size s           set the size of the sample to s
  --method m	     choose the clustering method m="kmeans" or m="dbscan"
  --onlypos b        choose if only position is considered (b=True) or not (b=False), i.e. also consider density
  --doint b          choose if integrals over clustering regions should be computed (b=True) or not (b=False)"
  --seed sd          choose seed for clustering, default is 0
  -v V               choose verbose mode, default is True
  --help             display this help and exit

Finally, the output of the NCICLUSTER program are a collection
of files : 
 - Two NCI 2D plots, one for all the data and one for data 
   below the isovalue.
 - A heatmap plot, that is supposed to guide the user in the
   adjustment of parameters, if needed.
 - One cube file, system1-cli-grad.cube, for each i-th cluster
   found, containing the gradient and subgrid for each separate
   cluster.
 - One cube file, system1-cli-dens.cube, for each i-th cluster
   found, containing the density and subgrid for each separate
   cluster.
 - One data file, system1-cli.dat, for each i-th cluster found,
   containing the density*sign(lambda_2) and the RDG.
 - A .vmd file.
   
There is also a standard output message that keeps the user 
informed on the progression of the computations. 
