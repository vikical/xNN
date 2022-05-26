# xNN
TODO: description


# REQUIREMENTS
Before running the programm, install **python=3.8** and the required libraries listed in **requirements.txt** file. A environment manager is suggested (i.e. conda)
~~~
# OPTIONAL: Environment creation (with conda).
conda create -n xnn_env python=3.8
# OPTIONAL: Environment activation (with conda).
conda activate xnn_env

# Dependencies installed.
pip install -r requirements.txt
~~~

Note that the code is optimized for:

- python=3.8
- tensorflow=2.3
- cuda=10.1

To change this requirement to adjust the code to different environments, please, take into account the following installation guide: https://www.tensorflow.org/install/source#gpu.

# RUNNING ONE SINGLE INSTANCE
TODO: update
To run the problem getting on one single instance, use the following command:
~~~
python app.py solve --din *path_to_directory* --problem *problem_number* --metaheu *type_of_metaheuristic* --niter *iterations_per_search* --init *type_initialization* --memory *memory_size_for_tabu_search* --maxs *max_running_seconds* --log *type_of_debug*
~~~
Where the options are:

**--din**: path to the directory where files with distances between nodes and for vehicles are located. This files should follow reg. exp. dist_NUMBER.txt and veh_dist_NUMBER.txt

**--problem**: Number of the problem instance to be solved, i.e. the NUMBER in the dist_NUMBER.txt files. By default: 0.

**--metaheu**: comma-separed list of metaheuristics which are going to be compare. Options: ls (local search); ts (tabu search); vnd; bvns

**--niter**: Number of iters per each local search. By default 10.

**--search**: decides between greedy search ('greedy') or anxious search ('anxious'). By default: greedy.

**--init**: dedices whether to init randomly ('random'), one customer to one client ('one2one'), a group of nodes ('group'). By default: random.

**--memory**: length of the memory for TS, otherwise ignored. By default 0.

**--maxs**: max. seconds an algorithm will run. By default 3.0s.

**--log**: Log level: DEBUG, INFO, CRITICAL. By default: CRITICAL.


The result of this execution will show the solution **cost** and the solution **routes**.

Example:
~~~
python app.py solve --din ./my_data --problem 0 --metaheu bvns --niter 10 --search anxious --init random --maxs 5 --log INFO
python app.py solve --din ./my_data --problem 0 --metaheu ts --niter 10 --search anxious --init random --memory 15 --maxs 5 --log CRITICAL
~~~

# RUNNING TEST BENCH
To run a set of problems and obtain a .csv the following command should be executed:
~~~
python app.py testbench --din *path_to_directory* --metaheu *comma_separated_list_of_metaheu* --niter *iterations_per_search* --search *type_of_search*
--init *type_initialization* --memory *memory_size_for_tabu_search* --maxs *max_running_seconds* --times4ave *times_for_average* --log *type_of_logging*
~~~
Where the options are:

**--din**: path to the directory where files with distances between nodes and for vehicles are located. This files should follow reg. exp. dist_NUMBER.txt and veh_dist_NUMBER.txt

**--metaheu**: comma-separed list of metaheuristics which are going to be compare. Options: ls (local search); ts (tabu search); vnd; bvns

**--niter**: Number of iters per each local search. By default 10.

**--search**: decides between greedy search ('greedy') or anxious search ('anxious').

**--init**: dedices whether to init randomly ('random'), one customer to one client ('one2one'), a group of nodes ('group'). By default: random.

**--memory**: length of the memory for TS, otherwise ignored. By default 0.

**--maxs**: max. seconds an algorithm will run. By default 3.0s.

**--times4ave**: number of times a problem instances is solved, to get an average of the solutions

**--log**: Log level: DEBUG, INFO, CRITICAL. By default: CRITICAL.



Example:
~~~
python app.py testbench --din ./my_data/30nodes --metaheu ls,vnd,bvns,ts --niter 10 --search greedy  --init group --memory 10  --maxs 1 --times4ave 100 --log CRITICAL
~~~

The result of this execution will be a .csv, stored in *path_to_directory* where each column will be the values obtained after *time_for_average* executions of each metaheuristic:
~~~
LS_average_execution_time ; LS_average_cost ; LS_best_cost ; VND_average_execution_time ; VND_average_cost ; VND_best_cost ; ...
... BVNS_average_execution_time; BVNS_average_cost; BVNS_best_cost; Tabu_average_execution_time; Tabu_average_cost ; Tabu_best_cost 
~~~

There are two scripts to run several examples at once. For memory size analysis: **test_bench_ts.sh**. For metaheuristics comparison: **test_bench.sh**.
In the directory ./results the output files of the execution of this script can be found. Apart from that, individual .ods files have been added, where graphs based on the obtained info are displayed. A **Summary.ods** gathers all this information and add new graphs comparing results between executions with different parameters.

# TEST
From main folder invoke:
python -m unittest discover test