import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import csv
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import os
from numpy import array
from sklearn.cross_validation import KFold
import math
import pandas as pd


model_name = 'DKT-DSC'
data_name= 'CAT' # 



# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate",1e-2, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.6, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("hidden_size", 200, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("problem_len", 20, "length for each time interval")
tf.flags.DEFINE_integer("num_cluster", 8, "length for each time interval")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("train_data_path", 'data/'+data_name+'_train.csv', "Path to the training dataset")
tf.flags.DEFINE_string("test_data_path", 'data/'+data_name+'_test.csv', "Path to the testing dataset")
tf.flags.DEFINE_boolean("model_name", model_name, "model used")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def add_gradient_noise(t, stddev=1e-3, name=None):

    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class StudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self.num_skills = num_skills = config.num_skills        
        self.hidden_size = size = FLAGS.hidden_size
        self.num_steps = num_steps = config.num_steps
        input_size = (num_skills*2)
        output_size = num_skills # actual size batch_size * num_steps * output_size
        inputs = self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target_id = target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        final_hidden_size = size

        hidden_layers = []
        for i in range(FLAGS.hidden_layer_num):
            final_hidden_size = size/(i+1)
            hidden1 = tf.nn.rnn_cell.LSTMCell(final_hidden_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.nn.rnn_cell.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)

        cell = tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)
        input_data = tf.reshape(self._input_data, [-1])

        #one-hot encoding
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(input_data, 1)
       	    indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
            concated = tf.concat([indices, labels],1)
            inputs = tf.sparse_to_dense(concated, tf.stack([batch_size*num_steps, input_size]), 1.0, 0.0)
            inputs.set_shape([batch_size*num_steps, input_size])
            
        inputs = tf.reshape(inputs, [-1, num_steps, input_size])
        x = tf.transpose(inputs, [1, 0, 2])
        x = tf.reshape(x, [-1, input_size])
        x = tf.split(x, num_steps, 0)
        outputs, state = rnn.static_rnn(cell, x, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs,1), [-1, int(final_hidden_size)])
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, output_size])
        sigmoid_b = tf.get_variable("sigmoid_b", [output_size])
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b
        logits = tf.reshape(logits, [-1])        
        selected_logits = tf.gather(logits, self.target_id)
        self._all_logits = logits

        #make prediction
        self._pred = tf.sigmoid(selected_logits)

        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def all_logits(self):
        return self._all_logits

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05    
    num_skills = 0
    num_steps = FLAGS.problem_len
    batch_size = FLAGS.batch_size
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
        



def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def k_means_clust(session, train_students, test_students, max_stu, max_seg, num_clust, num_iter,w=50):
    identifiers=2    
    max_stu=int(max_stu)
    max_seg=int(max_seg)
    cluster= np.zeros((max_stu,max_seg))
    data=[]
    for ind,i in enumerate(train_students):
        data.append(i[:-identifiers])
    data = array(data)
    points = tf.constant(data)    
    
    # choose random k points (k, 2)
    centroids = tf.Variable(tf.random_shuffle(points)[:num_clust, :])
    # calculate distances from the centroids to each point
    points_e = tf.expand_dims(points, axis=0) # (1, N, 2)
    centroids_e = tf.expand_dims(centroids, axis=1) # (k, 1, 2)
    distances = tf.reduce_sum((points_e - centroids_e) ** 2, axis=-1) # (k, N)
    # find the index to the nearest centroids from each point
    indices = tf.argmin(distances, axis=0) # (N,)
    # gather k clusters: list of tensors of shape (N_i, 1, 2) for each i
    clusters = [tf.gather(points, tf.where(tf.equal(indices, i))) for i in range(num_clust)]
    # get new centroids (k, 2)
    new_centroids = tf.concat([tf.reduce_mean(clusters[i], reduction_indices=[0]) for i in range(num_clust)], axis=0)
    # update centroids
    assign = tf.assign(centroids, new_centroids)
    session.run(tf.global_variables_initializer())
    for j in range(10):
        clusters_val, centroids_val, _ = session.run([clusters, centroids, assign])

    for ind,i in enumerate(train_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None            
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j],w)< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j],w)
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j
        cluster[int(i[-2]),int(i[-1])]=closest_clust

        
    
    for ind,i in enumerate(test_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None 
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j],w)< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j],w)
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j
        cluster[int(i[-2]),int(i[-1])]=closest_clust
    return cluster


    
    

def cluster_data(students,max_stu,num_skills):

    success = []
    max_seg =0    
    xtotal = np.zeros((max_stu,num_skills))
    x1 = np.zeros((max_stu,num_skills))
    x0 = np.zeros((max_stu,num_skills))
    index = 0  
    while(index+FLAGS.batch_size < len(students)):    
         for i in range(FLAGS.batch_size):
             student = students[index+i] 
             student_id = int(student[0][0])
             seg_id = int(student[0][1])
             if (int(student[0][3])==1):
                tmp_seg = seg_id
                if(tmp_seg > max_seg):
                   max_seg = tmp_seg
                problem_ids = student[1]                
                correctness = student[2]
                for j in range(len(problem_ids)):           
                    key =problem_ids[j]
                    xtotal[student_id,key] +=1
                    if(int(correctness[j]) == 1):
                      x1[student_id,key] +=1
                    else:
                         x0[student_id,key] +=1

                xsr=[x/y for x, y in zip(x1[student_id], xtotal[student_id])]
                xfr=[x/y for x, y in zip(x0[student_id], xtotal[student_id])]
             
                x=np.nan_to_num(xsr)-np.nan_to_num(xfr)
                x=np.append(x, student_id)
                x=np.append(x, seg_id)
                success.append(x) 
         index += FLAGS.batch_size  
    
    return success, max_seg 
    
    
  
def get_steps(config,train_students, test_students, cluster):      
    index = 0   
    max_index = config.num_steps * config.num_skills * config.batch_size
    max_t_indx= max_index     
    while(index+config.batch_size < len(train_students)):        
        for i in range(config.batch_size):
            student = train_students[index+i]
            student_id = student[0][0]
            seg_id = int(student[0][1])
            if (seg_id>0):
                cluster_id= cluster[student_id,(seg_id-1)]+2
            else:
                cluster_id= 1
            problem_ids = student[1]
            for j in range(len(problem_ids)-1):                
                target_indx = j+1 
                target_id = int(problem_ids[target_indx])
                # to ignore if target_id is null or -1 all skill index are started from 0
                if target_id > -1: 
                   burffer_space=i*config.num_steps*(config.num_skills)+j*(config.num_skills)
                   t_id=burffer_space+ int(target_id * cluster_id)
                   if(t_id>max_t_indx):
                      max_t_indx = t_id                               
        index += config.batch_size
    index = 0     
    while(index+config.batch_size < len(test_students)):        
        for i in range(config.batch_size):
            student = test_students[index+i]
            student_id = student[0][0]
            seg_id = int(student[0][1])
            if (seg_id>0):
                cluster_id= cluster[student_id,(seg_id-1)]+2
            else:
                cluster_id= 1
            problem_ids = student[1]
            for j in range(len(problem_ids)-1):                
                target_indx = j+1 
                target_id = int(problem_ids[target_indx])
                # to ignore if target_id is null or -1 all skill index are started from 0
                if target_id > -1: 
                   burffer_space=i*config.num_steps*(config.num_skills)+j*(config.num_skills)
                   t_id=burffer_space+ int(target_id * cluster_id)
                   if(t_id>max_t_indx):
                      max_t_indx = t_id                               
        index += config.batch_size        
    num_steps = (max_index // (config.num_skills * config.batch_size))+1
    if (num_steps<config.num_steps):
       num_steps=config.num_steps
    return num_steps
         


def run_epoch(session, m, students, cluster, eval_op, verbose=False):
    """Runs the model on the given data."""   
    index = 0
    pred_labels = []
    actual_labels = []
    all_all_logits = []
    
    while(index+m.batch_size < len(students)):
        x = np.zeros((m.batch_size, m.num_steps))
        target_ids = []
        target_correctness = []        
        for i in range(m.batch_size):
            student = students[index+i]
            student_id = student[0][0]
            seg_id = int(student[0][1])            

            ## assign cluster of student at segment z-1
            ## seg_id==0 is initial segment with initial unidentified cluster for all student
            if (seg_id>0):
                cluster_id= cluster[student_id,(seg_id-1)]+2
            else:
                cluster_id= 1            
            
            problem_ids = student[1]
            correctness = student[2]            
            #print ('*'*20+str(cluster_id))
            #print (problem_ids)
            #print (correctness)
                        
            for j in range(len(problem_ids)-1):
                current_indx= j 
                target_indx = j+1
                
                current_id = int(problem_ids[current_indx])
                target_id = int(problem_ids[target_indx])                
                label_index = 0
                correct = int(correctness[current_indx])
                # to ignore if target_id is null or -1 all skill index are started from 0
                if target_id > -1:  
                   if( correct == 0):
                      label_index = current_id 
                   else:
                       label_index =(current_id + m.num_skills)                
                   
                   x[i, j] = label_index
                   
                   
                   burffer_space=i*m.num_steps*(m.num_skills)+j*(m.num_skills)
                   t_ind=burffer_space+ int(target_id * cluster_id)
                   target_ids.append(t_ind)
                    
                   target_correctness.append(int(correctness[target_indx]))                
                   actual_labels.append(int(correctness[target_indx]))
                
        index += m.batch_size
        
        pred, _, all_logits = session.run([m.pred, eval_op, m.all_logits], feed_dict={
            m.input_data: x, m.target_id: target_ids,
            m.target_correctness: target_correctness})
        
        for i, p in enumerate(pred):
            pred_labels.append(p)

        all_all_logits.append(all_logits)
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_labels)

    return rmse, auc, r2, np.concatenate(all_all_logits)


def read_data_from_csv_file(fileName, shuffle=False):
    config = HyperParamsConfig()
    rows = []
    max_skills = 0
    max_steps = 0 
    studentids = []
    
    problem_len = FLAGS.problem_len    
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    
    index = 0   
    tuple_rows = []
    while(index < len(rows)-1):
          problems = int(rows[index][0]) 
          student_id= int(rows[index][1])
          studentids.append(student_id)
          if (problems>problem_len):
          
             tmp_max_steps = int(rows[index][0])
             if(tmp_max_steps > max_steps):
                max_steps = tmp_max_steps 
           
             tmp_max_skills = max(map(int, rows[index+1]))
             if(tmp_max_skills > max_skills):
                max_skills = tmp_max_skills
           
             len_problems = int(int(problems) / problem_len)*problem_len
             rest_problems = problems - len_problems             
             
             ele_p = []             
             p_index=0       
             for element in rows[index+1]:
                 ele_p.append(int(element)+1)
                 p_index=p_index+1 
             ele_c = []
             c_index=0
             for element in rows[index+2]:
                 ele_c.append(int(element))
                 c_index=c_index+1

             if (rest_problems>0):
                rest=problem_len-rest_problems
                for i in range(rest):
                    ele_p.append(-1)
                    ele_c.append(-1)

             ele_p_array = np.reshape(np.asarray(ele_p), (-1,problem_len))
             ele_c_array = np.reshape(np.asarray(ele_c), (-1,problem_len))
           
             n_pieces = ele_p_array.shape[0]
           
             for j in range(n_pieces):
                 s1=[student_id,j,problems]
                 if (j>-1) & (j< (n_pieces-1)) :
                    s1.append(1)
                    s2= np.append(ele_p_array[j,:],ele_p_array[j+1,0]).tolist()
                    s3= np.append(ele_c_array[j,:],ele_c_array[j+1,0]).tolist()      
                 else:
                      s1.append(-1)
                      s2= ele_p_array[j,:].tolist()
                      s3= ele_c_array[j,:].tolist() 
                 tup = (s1,s2,s3)
                 tuple_rows.append(tup)
             index += 3
          else:
               ele_p = [] 
               p_index=0       
               for element in rows[index+1]:
                   ele_p.append(int(element)+1)
                   p_index=p_index+1              
               
               ele_c = []
               c_index=0
               for element in rows[index+2]:
                   ele_c.append(int(element))
                   c_index=c_index+1
 
               rest=problem_len-problems
               for i in range(rest):
                   ele_p.append(-1)
                   ele_c.append(-1)
               s1=[student_id,0,problems]
               s1.append(-1)
               tup = (s1,ele_p,ele_c)
               tuple_rows.append(tup)         
               index += 3
             
    #print (studentid)           
    #print(studentid)
    
    print ("the number of skills is " + str(max_skills))
    print ("the Problem_steps in original is " + str(max_steps))
    print ("the Problem_steps after processing is " + str(problem_len)) 
    print ("Finish reading data")
    max_skills =max_skills+2
    max_steps  =max_steps+1
    return tuple_rows, studentids, max_skills 


def main(unused_args):
    config = HyperParamsConfig()
    
    train_students, train_ids, train_max_skills = read_data_from_csv_file(FLAGS.train_data_path, shuffle=True)
    test_students, test_ids, test_max_skills = read_data_from_csv_file(FLAGS.test_data_path, shuffle=True)    
    max_skills=max([int(train_max_skills),int(test_max_skills)])+1    
    config.num_skills = max_skills 
    train_cluster_data, train_max_seg= cluster_data(train_students,max(train_ids)+1,max_skills)
    test_cluster_data, test_max_seg= cluster_data(test_students,max(test_ids)+1,max_skills)
    max_stu= max(train_ids+test_ids)+1
    max_seg=max([int(train_max_seg),int(test_max_seg)])+1
    
    with tf.Graph().as_default():
         session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
         global_step = tf.Variable(0, name="global_step", trainable=False)
         starter_learning_rate = FLAGS.learning_rate
         learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)
         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
         
         with tf.Session(config=session_conf) as session:              
              cluster =k_means_clust(session, train_cluster_data, test_cluster_data, max_stu, max_seg, FLAGS.num_cluster, 10)
              config.num_steps = get_steps(config, train_students, test_students, cluster)
              initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
              # training model
              with tf.variable_scope("model", reuse=None, initializer=initializer):
                   m = StudentModel(is_training=True, config=config)
              # testing model
              with tf.variable_scope("model", reuse=True, initializer=initializer):
                   mtest = StudentModel(is_training=False, config=config)
              grads_and_vars = optimizer.compute_gradients(m.cost)
              grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
              for g, v in grads_and_vars if g is not None]
              grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
              train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
              session.run(tf.initialize_all_variables())
              j=1
              for i in range(config.max_max_epoch):
                  rmse, auc, r2, _ = run_epoch(session, m, train_students, cluster, train_op, verbose=False)
                  print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, auc, r2))
                  if((i+1) % FLAGS.evaluation_interval == 0):
                     rmse, auc, r2, all_logits = run_epoch(session, mtest, test_students, cluster, tf.no_op(), verbose=True)
                     print("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (j, rmse, auc, r2))
                     j+=1
                        
       
             
               
if __name__ == "__main__":
    tf.app.run()
