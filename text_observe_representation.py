"""
Text-based observation representation using SentenceTransformer
"""
import numpy as np
import hashlib
import pickle
import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


class TextObservationEncoder:
    """
    Encodes environment state into text descriptions and then to embeddings
    using SentenceTransformer with caching support (in-memory + persistent file)
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir="./text_cache"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        # In-memory cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Persistent cache configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache file with model hash to support multiple models
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        self.cache_file = self.cache_dir / f"embeddings_cache_{model_hash}.pkl"
        self.stats_file = self.cache_dir / f"cache_stats_{model_hash}.json"
        
        # Load existing cache from file
        self._load_cache_from_file()
        
        print(f"Initialized SentenceTransformer: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cache file: {self.cache_file.name}")
        print(f"Loaded {len(self.cache)} embeddings from cache")
    
    def _load_cache_from_file(self):
        """Load cache from persistent file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Successfully loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            else:
                print(f"Cache file not found, starting with empty cache")
                self.cache = {}
        except Exception as e:
            print(f"Error loading cache from file: {e}")
            print("Starting with empty cache")
            self.cache = {}
    
    def _save_cache_to_file(self):
        """Save cache to persistent file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving cache to file: {e}")
    
    def _save_stats_to_file(self):
        """Save cache statistics to JSON file"""
        try:
            stats = self.get_cache_stats()
            stats['cache_size_mb'] = self._get_cache_size_mb()
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats to file: {e}")
    
    def _get_cache_size_mb(self):
        """Get approximate cache size in MB"""
        total_size = 0
        for embedding in self.cache.values():
            total_size += embedding.nbytes
        return total_size / (1024 * 1024)
    
    def _get_cache_key(self, text):
        """Generate cache key from text description"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _encode_cached(self, texts):
        """
        Encode texts with caching. 
        Automatically saves cache to file after each batch of new encodings.
        
        Args:
            texts: list of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Separate cached and uncached texts
        cache_keys = [self._get_cache_key(text) for text in texts]
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self.cache:
                cached_embeddings[i] = self.cache[key]
                self.cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.cache_misses += 1
        
        # Encode uncached texts
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts)
            for idx, new_emb, text in zip(uncached_indices, new_embeddings, uncached_texts):
                key = self._get_cache_key(text)
                self.cache[key] = new_emb
                embeddings[idx] = new_emb
            
            # Save cache to file after new encodings
            self._save_cache_to_file()
        
        # Fill in cached embeddings
        for idx, emb in cached_embeddings.items():
            embeddings[idx] = emb
        
        return embeddings
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def print_cache_stats(self):
        """Print detailed cache statistics"""
        stats = self.get_cache_stats()
        print("\n" + "="*50)
        print("Cache Statistics:")
        print("="*50)
        print(f"Cache entries: {stats['cache_size']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Hit rate: {stats['hit_rate']:.2f}%")
        print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"Cache file: {self.cache_file}")
        print("="*50 + "\n")
    
    def clear_cache(self):
        """Clear both in-memory and persistent cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        try:
            if self.cache_file.exists():
                os.remove(self.cache_file)
            if self.stats_file.exists():
                os.remove(self.stats_file)
            print(f"Cache cleared and cache files deleted")
        except Exception as e:
            print(f"Error clearing cache files: {e}")
    
    def save_cache_checkpoint(self, name="checkpoint"):
        """Save cache to a checkpoint file for backup"""
        try:
            checkpoint_file = self.cache_dir / f"cache_checkpoint_{name}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._save_stats_to_file()
            print(f"Cache checkpoint saved to {checkpoint_file}")
            print(f"Cache size: {self._get_cache_size_mb():.2f} MB")
            return checkpoint_file
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def load_cache_checkpoint(self, name="checkpoint"):
        """Load cache from a checkpoint file"""
        try:
            checkpoint_file = self.cache_dir / f"cache_checkpoint_{name}.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Cache loaded from checkpoint {checkpoint_file}")
                print(f"Loaded {len(self.cache)} embeddings")
                return True
            else:
                print(f"Checkpoint file not found: {checkpoint_file}")
                return False
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def cleanup_cache(self, max_size_mb=500):
        """
        Cleanup cache if it exceeds max size.
        Keeps most frequently used entries.
        """
        current_size = self._get_cache_size_mb()
        if current_size > max_size_mb:
            print(f"Cache size {current_size:.2f} MB exceeds limit {max_size_mb} MB. Cleaning up...")
            
            # Sort by access frequency (we track hits per key)
            # For now, keep the most recent 80% of entries
            num_to_keep = int(len(self.cache) * 0.8)
            
            # Sort by key (hash) to get deterministic behavior
            sorted_keys = sorted(self.cache.keys())
            keys_to_remove = sorted_keys[:len(sorted_keys) - num_to_keep]
            
            for key in keys_to_remove:
                del self.cache[key]
            
            self._save_cache_to_file()
            
            new_size = self._get_cache_size_mb()
            print(f"Cache cleaned up. New size: {new_size:.2f} MB. Entries removed: {len(keys_to_remove)}")
    
    def get_cache_dir_size(self):
        """Get total size of cache directory in MB"""
        total_size = 0
        for file in self.cache_dir.glob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)
    
    def encode_state(self, env):
        """
        Convert environment state to text descriptions and encode to embeddings
        Each part is encoded separately then concatenated.
        Uses caching to avoid re-encoding identical descriptions.
        
        Args:
            env: Environment object
            
        Returns:
            numpy array of concatenated embeddings (1, total_dim)
        """
        embeddings_list = []
        
        # 1. Encode cluster resources
        resource_desc = self._describe_resources(env)
        resource_embs = self._encode_cached([resource_desc])  # Shape: (1, embedding_dim)
        embeddings_list.append(resource_embs[0])
        
        # 2. Encode job slots (visible new jobs)
        job_descs = []
        for i in range(env.pa.num_nw):
            job_desc = self._describe_job_slot(env, i)
            job_descs.append(job_desc)
        
        if job_descs:
            job_embs = self._encode_cached(job_descs)  # Shape: (num_nw, embedding_dim)
            for job_emb in job_embs:
                embeddings_list.append(job_emb)
        
        # 3. Encode backlog
        backlog_desc = self._describe_backlog(env)
        backlog_embs = self._encode_cached([backlog_desc])  # Shape: (1, embedding_dim)
        embeddings_list.append(backlog_embs[0])
        
        # 4. Encode running jobs
        running_desc = self._describe_running_jobs(env)
        running_embs = self._encode_cached([running_desc])  # Shape: (1, embedding_dim)
        embeddings_list.append(running_embs[0])
        
        # 5. Encode temporal information
        temporal_desc = self._describe_temporal_info(env)
        temporal_embs = self._encode_cached([temporal_desc])  # Shape: (1, embedding_dim)
        embeddings_list.append(temporal_embs[0])
        
        # Concatenate all embeddings
        state = np.concatenate(embeddings_list)  # Shape: (total_parts, embedding_dim) -> flattened
        
        return state[np.newaxis, :]  # Shape: (1, total_dim)
    
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
def compute_text_feature_dim(encoder, num_nw):
    """
    Compute dimension for text-based representation
    
    Breakdown of text parts:
    - 1 cluster resource description (1 embedding)
    - num_nw job slot descriptions (num_nw embeddings)
    - 1 backlog description (1 embedding)
    - 1 running jobs description (1 embedding)
    - 1 temporal description (1 embedding)
    
    Total parts: 1 + num_nw + 1 + 1 + 1 = num_nw + 4
    Each part is a 384-dimensional embedding (for all-MiniLM-L6-v2)
    
    Total feature dimension = (num_nw + 4) * 384
    """
    num_parts = 1 + num_nw + 1 + 1 + 1  # cluster + slots + backlog + running + temporal
    total_dim = num_parts * encoder.embedding_dim
    
    print(f"Text representation dimension breakdown:")
    print(f"  - 1 cluster resource part: {encoder.embedding_dim}D")
    print(f"  - {num_nw} job slot parts: {num_nw} × {encoder.embedding_dim}D")
    print(f"  - 1 backlog part: {encoder.embedding_dim}D")
    print(f"  - 1 running jobs part: {encoder.embedding_dim}D")
    print(f"  - 1 temporal part: {encoder.embedding_dim}D")
    print(f"  Total parts: {num_parts}")
    print(f"  Total dimension: {total_dim}D")
    
    return total_dim