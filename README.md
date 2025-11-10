# DeepRM
HotNets'16 http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf

Install prerequisites

```
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user Theano
pip install --user Lasagne==0.1
sudo apt-get install python-matplotlib
```

In folder RL, create a data/ folder. 

Use `launcher.py` to launch experiments. 


```
--exp_type <type of experiment> 
--num_res <number of resources> 
--num_nw <number of visible new work> 
--simu_len <simulation length> 
--num_ex <number of examples> 
--num_seq_per_batch <rough number of samples in one batch update> 
--eps_max_len <episode maximum length (terminated at the end)>
--num_epochs <number of epoch to do the training>
--time_horizon <time step into future, screen height> 
--res_slot <total number of resource slots, screen width> 
--max_job_len <maximum new job length> 
--max_job_size <maximum new job resource request> 
--new_job_rate <new job arrival rate> 
--dist <discount factor> 
--lr_rate <learning rate> 
--ba_size <batch size> 
--pg_re <parameter file for pg network> 
--v_re <parameter file for v network> 
--q_re <parameter file for q network> 
--out_freq <network output frequency> 
--ofile <output file name> 
--log <log file name> 
--render <plot dynamics> 
--unseen <generate unseen example> 
```


The default variables are defined in `parameters.py`.


Example: 
  - launch policy gradient with state representation using image-like structure
  
  ```
  python launcher.py --exp_type=pg_re --simu_len=50 --num_ex=10 --ofile=data/pg_image --repre=image > log_pg_image.txt
  ```
  - launch policy gradient with state representation using resource and job slot vectors
  
  ```
  python launcher.py --exp_type=pg_re --simu_len=50 --num_ex=10 --ofile=data/pg_feat --repre=feat_extract > log_pg_feat.txt
  ```
  - launch policy gradient with state representation using text description
  
  ```
  python launcher.py --exp_type=pg_re --simu_len=50 --num_ex=10 --ofile=data/pg_text --repre=text > log_pg_text.txt
  ```
