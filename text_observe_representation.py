"""
Text-based observation representation using SentenceTransformer
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class TextObservationEncoder:
    """
    Encodes environment state into text descriptions and then to embeddings
    using SentenceTransformer
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Initialized SentenceTransformer: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_state(self, env):
        """
        Convert environment state to text descriptions and encode to embeddings
        
        Args:
            env: Environment object
            
        Returns:
            numpy array of embeddings
        """
        descriptions = []
        
        # 1. Describe cluster resources
        resource_desc = self._describe_resources(env)
        descriptions.append(resource_desc)
        
        # 2. Describe job slots (visible new jobs)
        for i in range(env.pa.num_nw):
            job_desc = self._describe_job_slot(env, i)
            descriptions.append(job_desc)
        
        # 3. Describe backlog
        backlog_desc = self._describe_backlog(env)
        descriptions.append(backlog_desc)
        
        # 4. Describe running jobs
        running_desc = self._describe_running_jobs(env)
        descriptions.append(running_desc)
        
        # 5. Describe temporal information
        temporal_desc = self._describe_temporal_info(env)
        descriptions.append(temporal_desc)
        
        # Encode all descriptions
        embeddings = self.model.encode(descriptions)
        
        # Flatten and return
        return embeddings.flatten()[np.newaxis, :]
    
    def _describe_resources(self, env):
        """Describe cluster resource state"""
        desc_parts = []
        
        for i in range(env.pa.num_res):
            total_slots = env.pa.res_slot * env.pa.time_horizon
            available = np.sum(env.machine.avbl_slot[:, i])
            used = total_slots - available
            
            utilization_pct = (used / total_slots) * 100
            
            desc_parts.append(
                f"Resource {i}: {available} slots free out of {total_slots} total "
                f"({utilization_pct:.1f}% utilized)"
            )
        
        # Overall cluster status
        avg_util = np.mean([
            (env.pa.res_slot * env.pa.time_horizon - np.sum(env.machine.avbl_slot[:, i])) / 
            (env.pa.res_slot * env.pa.time_horizon) * 100
            for i in range(env.pa.num_res)
        ])
        
        if avg_util < 30:
            status = "lightly loaded"
        elif avg_util < 70:
            status = "moderately loaded"
        else:
            status = "heavily loaded"
        
        cluster_desc = f"Cluster is {status} with average {avg_util:.1f}% utilization. " + " ".join(desc_parts)
        
        return cluster_desc
    
    def _describe_job_slot(self, env, slot_idx):
        """Describe a job in the job slot"""
        job = env.job_slot.slot[slot_idx]
        
        if job is None:
            return f"Job slot {slot_idx} is empty and available for new jobs."
        
        # Describe resource requirements
        res_reqs = []
        for i in range(env.pa.num_res):
            if job.res_vec[i] > 0:
                res_reqs.append(f"{job.res_vec[i]} units of resource {i}")
        
        res_desc = ", ".join(res_reqs)
        
        # Check if job can be scheduled now
        can_schedule = self._can_schedule_job(env, job)
        schedule_status = "can be scheduled immediately" if can_schedule else "cannot be scheduled yet"
        
        # Calculate waiting time
        wait_time = env.curr_time - job.enter_time
        
        # Total resource demand
        total_demand = np.sum(job.res_vec)
        
        desc = (
            f"Job slot {slot_idx}: Job requires {res_desc}, "
            f"needs {job.len} time units to process, "
            f"total resource demand is {total_demand} units. "
            f"This job has been waiting for {wait_time} time units and {schedule_status}."
        )
        
        return desc
    
    def _describe_backlog(self, env):
        """Describe the backlog queue"""
        if env.job_backlog.curr_size == 0:
            return "Backlog queue is empty with no waiting jobs."
        
        # Calculate statistics
        total_res_demand = 0
        total_len = 0
        job_count = 0
        
        for job in env.job_backlog.backlog:
            if job is not None:
                total_res_demand += np.sum(job.res_vec)
                total_len += job.len
                job_count += 1
        
        if job_count > 0:
            avg_res_demand = total_res_demand / job_count
            avg_len = total_len / job_count
        else:
            avg_res_demand = 0
            avg_len = 0
        
        capacity_pct = (env.job_backlog.curr_size / env.pa.backlog_size) * 100
        
        if capacity_pct < 30:
            pressure = "low"
        elif capacity_pct < 70:
            pressure = "moderate"
        else:
            pressure = "high"
        
        desc = (
            f"Backlog has {env.job_backlog.curr_size} jobs waiting "
            f"({capacity_pct:.1f}% of capacity), indicating {pressure} system pressure. "
            f"Average job requires {avg_res_demand:.1f} resource units and "
            f"{avg_len:.1f} time units to complete."
        )
        
        return desc
    
    def _describe_running_jobs(self, env):
        """Describe currently running jobs"""
        num_running = len(env.machine.running_job)
        
        if num_running == 0:
            return "No jobs are currently running in the cluster."
        
        # Calculate average remaining time
        total_remaining = 0
        for job in env.machine.running_job:
            remaining = job.finish_time - env.curr_time
            total_remaining += remaining
        
        avg_remaining = total_remaining / num_running
        
        # Resource consumption
        res_usage = []
        for i in range(env.pa.num_res):
            used = 0
            for job in env.machine.running_job:
                if job.res_vec[i] > 0:
                    used += job.res_vec[i]
            res_usage.append(f"{used} units of resource {i}")
        
        res_desc = ", ".join(res_usage)
        
        desc = (
            f"{num_running} jobs are currently running in the cluster, "
            f"consuming {res_desc}. "
            f"Average remaining execution time is {avg_remaining:.1f} time units."
        )
        
        return desc
    
    def _describe_temporal_info(self, env):
        """Describe temporal information"""
        time_since_new = env.extra_info.time_since_last_new_job
        max_track = env.extra_info.max_tracking_time_since_last_job
        
        if time_since_new == 0:
            recency = "just arrived"
        elif time_since_new < max_track * 0.3:
            recency = "recently arrived"
        elif time_since_new < max_track * 0.7:
            recency = "not arrived for a while"
        else:
            recency = "not arrived for a long time"
        
        progress_pct = (env.seq_idx / env.pa.simu_len) * 100
        
        desc = (
            f"Last new job {recency} ({time_since_new} time units ago). "
            f"Current simulation is {progress_pct:.1f}% complete."
        )
        
        return desc
    
    def _can_schedule_job(self, env, job):
        """Check if a job can be scheduled immediately"""
        for t in range(0, env.pa.time_horizon - job.len):
            new_avbl_res = env.machine.avbl_slot[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                return True
        return False


# Update parameters.py to include text representation dimension
def compute_text_feature_dim(encoder):
    """Compute dimension for text-based representation"""
    # Number of text descriptions:
    # 1 cluster resource description
    # + num_nw job slot descriptions  
    # + 1 backlog description
    # + 1 running jobs description
    # + 1 temporal description
    # Each encoded to embedding_dim dimensions
    
    num_descriptions = 1 + encoder.pa.num_nw + 1 + 1 + 1
    return num_descriptions * encoder.embedding_dim