"""
Text-based observation representation using SentenceTransformer (all-mpnet-base-v2)
Each component is encoded as a single comprehensive prompt without caching
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class TextObservationEncoder:
    """
    Encodes environment state into text descriptions and embeddings.
    Uses all-mpnet-base-v2 model (768-dimensional embeddings).
    Each state component is encoded as one comprehensive prompt.
    """
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        print(f"Initialized TextObservationEncoder")
        print(f"Model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}D")
    
    def encode_state(self, env):
        """
        Convert environment state to text descriptions and encode to embeddings.
        
        State is encoded as 4 separate prompts:
        1. Cluster resources (overall resource state)
        2. Job queue (all visible job slots combined)
        3. Backlog and running jobs (system load)
        4. Temporal information (time context)
        
        Returns:
            numpy array of concatenated embeddings (1, 4 * embedding_dim)
        """
        # Collect all text descriptions
        prompts = []
        
        # 1. Cluster resources prompt
        cluster_prompt = self._create_cluster_prompt(env)
        prompts.append(cluster_prompt)
        
        # 2. Job queue prompt (all visible jobs)
        job_queue_prompt = self._create_job_queue_prompt(env)
        prompts.append(job_queue_prompt)
        
        # 3. Backlog and running jobs prompt
        system_load_prompt = self._create_system_load_prompt(env)
        prompts.append(system_load_prompt)
        
        # 4. Temporal information prompt
        temporal_prompt = self._create_temporal_prompt(env)
        prompts.append(temporal_prompt)
        
        # Encode all prompts at once (batch encoding for efficiency)
        embeddings = self.model.encode(prompts, show_progress_bar=False)
        
        # Concatenate embeddings: (4, embedding_dim) -> (4 * embedding_dim,)
        state = embeddings.flatten()
        
        return state[np.newaxis, :]  # Shape: (1, 4 * embedding_dim)
    
    def _create_cluster_prompt(self, env):
        """
        Create comprehensive prompt for cluster resource state.
        Describes overall capacity, utilization, and availability for all resources.
        """
        parts = ["Cluster Resource Status:"]
        
        # Overall cluster statistics
        total_resources = env.pa.num_res
        total_capacity_per_res = env.pa.res_slot * env.pa.time_horizon
        
        utilizations = []
        for i in range(env.pa.num_res):
            available = np.sum(env.machine.avbl_slot[:, i])
            used = total_capacity_per_res - available
            utilization_pct = (used / total_capacity_per_res) * 100
            utilizations.append(utilization_pct)
            
            parts.append(
                f"Resource {i}: {used}/{total_capacity_per_res} slots used "
                f"({utilization_pct:.1f}% utilized), {available} slots available"
            )
        
        # Overall status
        avg_util = np.mean(utilizations)
        if avg_util < 30:
            status = "lightly loaded with plenty of capacity"
        elif avg_util < 60:
            status = "moderately loaded with adequate capacity"
        elif avg_util < 85:
            status = "heavily loaded with limited capacity"
        else:
            status = "nearly saturated with very limited capacity"
        
        parts.insert(1, f"The cluster has {total_resources} resource types and is {status}.")
        
        # Resource balance
        if len(utilizations) > 1:
            util_std = np.std(utilizations)
            if util_std < 15:
                balance = "well-balanced across all resource types"
            elif util_std < 30:
                balance = "moderately imbalanced across resource types"
            else:
                balance = "highly imbalanced with some resources bottlenecked"
            parts.append(f"Resource utilization is {balance}.")
        
        return " ".join(parts)
    
    def _create_job_queue_prompt(self, env):
        """
        Create comprehensive prompt for job queue (visible job slots).
        Describes all pending jobs in the queue.
        """
        parts = ["Job Queue Status:"]
        
        # Count jobs by state
        total_slots = env.pa.num_nw
        empty_slots = sum(1 for slot in env.job_slot.slot if slot is None)
        occupied_slots = total_slots - empty_slots
        
        if occupied_slots == 0:
            return "Job Queue Status: The job queue is completely empty with all slots available for new jobs."
        
        parts.append(f"{occupied_slots} out of {total_slots} slots are occupied.")
        
        # Analyze jobs in queue
        job_descriptions = []
        schedulable_count = 0
        total_wait_time = 0
        resource_demands = {i: [] for i in range(env.pa.num_res)}
        job_lengths = []
        
        for i, job in enumerate(env.job_slot.slot):
            if job is None:
                continue
            
            # Collect statistics
            can_schedule = self._can_schedule_job(env, job)
            if can_schedule:
                schedulable_count += 1
            
            wait_time = env.curr_time - job.enter_time
            total_wait_time += wait_time
            job_lengths.append(job.len)
            
            # Resource requirements
            res_reqs = []
            for res_i in range(env.pa.num_res):
                if job.res_vec[res_i] > 0:
                    res_reqs.append(f"{job.res_vec[res_i]} units of resource {res_i}")
                    resource_demands[res_i].append(job.res_vec[res_i])
            
            schedule_status = "ready to schedule" if can_schedule else "blocked waiting for resources"
            
            job_descriptions.append(
                f"Slot {i}: requires {', '.join(res_reqs)}, "
                f"duration {job.len} time units, "
                f"waiting {wait_time} time units, {schedule_status}"
            )
        
        # Add job descriptions
        if job_descriptions:
            parts.append("Jobs in queue: " + "; ".join(job_descriptions) + ".")
        
        # Summary statistics
        if occupied_slots > 0:
            avg_wait = total_wait_time / occupied_slots
            avg_len = np.mean(job_lengths)
            
            parts.append(
                f"Average job has waited {avg_wait:.1f} time units "
                f"and needs {avg_len:.1f} time units to complete."
            )
            
            if schedulable_count == 0:
                parts.append("No jobs can currently be scheduled due to resource constraints.")
            elif schedulable_count == occupied_slots:
                parts.append("All jobs in queue can be scheduled immediately.")
            else:
                parts.append(
                    f"{schedulable_count} out of {occupied_slots} jobs can be scheduled immediately."
                )
        
        return " ".join(parts)
    
    def _create_system_load_prompt(self, env):
        """
        Create comprehensive prompt for system load (backlog + running jobs).
        Describes both queued work and active work.
        """
        parts = ["System Load:"]
        
        # Backlog analysis
        backlog_size = env.job_backlog.curr_size
        backlog_capacity = env.pa.backlog_size
        
        if backlog_size == 0:
            backlog_desc = "The backlog is empty"
        else:
            backlog_pct = (backlog_size / backlog_capacity) * 100
            
            # Calculate backlog statistics
            total_res_demand = 0
            total_len = 0
            for job in env.job_backlog.backlog:
                if job is not None:
                    total_res_demand += np.sum(job.res_vec)
                    total_len += job.len
            
            avg_res = total_res_demand / backlog_size if backlog_size > 0 else 0
            avg_len = total_len / backlog_size if backlog_size > 0 else 0
            
            if backlog_pct < 30:
                pressure = "low pressure"
            elif backlog_pct < 70:
                pressure = "moderate pressure"
            else:
                pressure = "high pressure"
            
            backlog_desc = (
                f"The backlog contains {backlog_size} jobs ({backlog_pct:.0f}% full) "
                f"indicating {pressure}, with average job requiring {avg_res:.1f} resource units "
                f"and {avg_len:.1f} time units"
            )
        
        parts.append(backlog_desc + ".")
        
        # Running jobs analysis
        num_running = len(env.machine.running_job)
        
        if num_running == 0:
            running_desc = "No jobs are currently running"
        else:
            # Calculate running job statistics
            total_remaining = 0
            resource_usage = {i: 0 for i in range(env.pa.num_res)}
            
            for job in env.machine.running_job:
                remaining = job.finish_time - env.curr_time
                total_remaining += remaining
                for i in range(env.pa.num_res):
                    resource_usage[i] += job.res_vec[i]
            
            avg_remaining = total_remaining / num_running
            
            res_usage_strs = [
                f"{resource_usage[i]} units of resource {i}"
                for i in range(env.pa.num_res) if resource_usage[i] > 0
            ]
            
            running_desc = (
                f"{num_running} jobs are currently executing, "
                f"consuming {', '.join(res_usage_strs)}, "
                f"with average {avg_remaining:.1f} time units remaining"
            )
        
        parts.append(running_desc + ".")
        
        # Overall system pressure
        total_pending = backlog_size + sum(1 for s in env.job_slot.slot if s is not None)
        if total_pending == 0 and num_running == 0:
            overall = "The system is idle with no workload"
        elif total_pending < 5 and num_running < 3:
            overall = "The system has light workload"
        elif total_pending < 15 or num_running < 10:
            overall = "The system has moderate workload"
        else:
            overall = "The system has heavy workload with significant queuing"
        
        parts.append(overall + ".")
        
        return " ".join(parts)
    
    def _create_temporal_prompt(self, env):
        """
        Create comprehensive prompt for temporal information.
        Describes timing context and simulation progress.
        """
        parts = ["Temporal Context:"]
        
        # Job arrival pattern
        time_since_new = env.extra_info.time_since_last_new_job
        max_track = env.extra_info.max_tracking_time_since_last_job
        
        if time_since_new == 0:
            arrival_desc = "A new job just arrived in the current time step"
        elif time_since_new == 1:
            arrival_desc = "A new job arrived 1 time step ago"
        elif time_since_new < max_track * 0.3:
            arrival_desc = f"A new job arrived recently ({time_since_new} time steps ago)"
        elif time_since_new < max_track * 0.6:
            arrival_desc = f"No new jobs have arrived for {time_since_new} time steps (moderate idle period)"
        else:
            arrival_desc = f"No new jobs have arrived for {time_since_new} time steps (extended idle period)"
        
        parts.append(arrival_desc + ".")
        
        # Simulation progress
        progress_pct = (env.seq_idx / env.pa.simu_len) * 100
        
        if progress_pct < 10:
            phase = "early phase"
        elif progress_pct < 40:
            phase = "ramp-up phase"
        elif progress_pct < 70:
            phase = "mid phase"
        elif progress_pct < 90:
            phase = "late phase"
        else:
            phase = "final phase"
        
        parts.append(
            f"The simulation is {progress_pct:.0f}% complete, currently in {phase}."
        )
        
        # Current time context
        parts.append(f"Current simulation time is {env.curr_time}.")
        
        return " ".join(parts)
    
    def _can_schedule_job(self, env, job):
        """Check if a job can be scheduled immediately"""
        for t in range(0, env.pa.time_horizon - job.len):
            new_avbl_res = env.machine.avbl_slot[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                return True
        return False


def compute_text_feature_dim(encoder):
    """
    Compute dimension for text-based representation.
    
    With all-mpnet-base-v2 model:
    - Embedding dimension: 768D
    - Number of prompts: 4
      1. Cluster resources
      2. Job queue
      3. System load (backlog + running)
      4. Temporal information
    
    Total dimension = 4 * 768 = 3072
    """
    num_prompts = 4
    total_dim = num_prompts * encoder.embedding_dim
    
    print(f"\n{'='*60}")
    print(f"Text Representation Dimension Breakdown")
    print(f"{'='*60}")
    print(f"Model: {encoder.model_name}")
    print(f"Embedding dimension per prompt: {encoder.embedding_dim}D")
    print(f"\nPrompt structure:")
    print(f"  1. Cluster resources       : {encoder.embedding_dim}D")
    print(f"  2. Job queue (all slots)   : {encoder.embedding_dim}D")
    print(f"  3. System load (backlog+run): {encoder.embedding_dim}D")
    print(f"  4. Temporal information    : {encoder.embedding_dim}D")
    print(f"\nTotal prompts: {num_prompts}")
    print(f"Total dimension: {total_dim}D ({num_prompts} × {encoder.embedding_dim}D)")
    print(f"{'='*60}\n")
    
    return total_dim


# Test the implementation
if __name__ == '__main__':
    print("Testing TextObservationEncoder with all-mpnet-base-v2")
    print("=" * 70)
    
    # Create encoder
    encoder = TextObservationEncoder()
    
    # Compute and display dimension
    total_dim = compute_text_feature_dim(encoder)
    
    print("\nKey Features:")
    print("✓ No caching - direct encoding every time")
    print("✓ 4 comprehensive prompts covering all state aspects")
    print("✓ Each prompt provides rich contextual information")
    print("✓ all-mpnet-base-v2: 768D embeddings (higher quality than MiniLM)")
    print("✓ Total state representation: 3072D")
    
    print("\n" + "=" * 70)
    print("Implementation complete!")