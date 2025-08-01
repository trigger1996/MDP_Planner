P_MDP_TG
========

Planner for Markov Decision Processes with Temporal Goals 

```
@ARTICLE{8272366,
  author={M. {Guo} and M. M. {Zavlanos}},
  journal={IEEE Transactions on Automatic Control}, 
  title={Probabilistic Motion Planning Under Temporal Tasks and Soft Constraints}, 
  year={2018},
  volume={63},
  number={12},
  pages={4051-4066},
  doi={10.1109/TAC.2018.2799561}}
```

-----
Installation
-----

```
sudo apt update

pip3 install rich ortools seaborn matplotlib networkx

sudo apt install python3-tk

pip3 install --upgrade --force-reinstall pillow

sudo apt install texlive-latex-base texlive-fonts-recommended texlive-latex-extra dvipng
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

**install ltl2ba-1.3.tar.gz ltl2dstar-0.5.4.tar.gz and spot in MDP_TG/**



-----
Installation of Fonts
-----

准备字体文件
下载字体文件，例如：EuclidFlex-Regular.ttf、EuclidFlex-Bold.ttf 等。


```
创建字体目录（如果还没有）：

mkdir -p ~/.fonts

复制字体文件进去：

cp /path/to/your/EuclidFlex-*.ttf ~/.fonts/

刷新字体缓存：

fc-cache -f -v

（可选）检查字体是否被系统识别：

fc-list | grep -i euclid
```

-----
Description
-----
This package contains the implementation for policy synthesis algorithms given a probabilistically-labeled Markov Decision Process (MDP) (as the robot motion model) and a Linear Temporal Logic (LTL) formula (as the robot task). It outputs a stationary  and finite-memory policy consists of plan prefix and plan suffix, such that the controlled robot behavior fulfills the task with a given lower-bounded risk and minimizes the expected total cost. 


<p align="center">  
  <img src="https://github.com/MengGuo/P_MDP_TG/blob/master/MDP_TG/figures/risk.png" width="600"/>
</p>



-----
Features
-----
* Allows probabilistic labels on MDP states.
* **Tunable** _trade-off_ between `risk` and `expected total cost` in the plan prefix.
* Linear programs for solving constrained stochastic shortest path (SSP).
* Optimization over **both** plan prefix and suffix.
* **Relaxed** policy generation for cases where no accepting end components (AECs) exist.
* Interface between LTL formula, Buchi Automaton, Deterministic Robin Automaton and NetworkX graph objects.
* Computing maximal accepting end components (MAEC) of MDPs.
* [New] Clean storage of product automaton via pickle, for translating later to [PRISM language](http://www.prismmodelchecker.org/manual/ThePRISMLanguage/Introduction), see [the interface](https://github.com/MengGuo/PRISM_interface).


```python
from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_full_plan

# construct your motion MDP
motion_mdp = Motion_MDP(node_dict, edge_dict, U, initial_node, initial_label)

# specify your LTL task
surv_task = "& G F a & G F b G F c"

# construct DRA 
dra = Dra(surv_task)

# construct product DRA and accepting pairs
prod_dra = Product_Dra(motion_mdp, dra)
prod_dra.compute_S_f()

# policy synthesis 
allowed_risk = 0.1
best_all_plan = syn_full_plan(prod_dra, allowed_risk)
```

<p align="center">  
  <img src="https://github.com/MengGuo/P_MDP_TG/blob/master/MDP_TG/figures/mdp_tg.png" width="600"/>
</p>

* [New] Virtual experimental platform based on [V_REP](http://www.coppeliarobotics.com).

<p align="center">  
  <img src="https://github.com/MengGuo/P_MDP_TG/blob/master/v_rep/vrep.png" width="600"/>
</p>

----
Dependence
----
* Install python packages like Networkx, ply
* Compile [libspot](https://spot.lre.epita.fr/install.html) for your OS.
* Compile [ltl2ba executable](http://www.lsv.ens-cachan.fr/%7Egastin/ltl2ba/download.php) for your OS.
* Compile [ltl2dstar executable](http://www.ltl2dstar.de) for your OS. 
* [Gurobi solver](http://www.gurobi.com) for linear programs. Free for academic use.