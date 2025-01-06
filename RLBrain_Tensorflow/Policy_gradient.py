import numpy as np
import tensorflow as tf
from tensorflow import keras

# reproducible
np.random.seed(1)
tf.random.set_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = 0.95

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            self.summary_writer = tf.summary.create_file_writer("logs/")

    def _build_net(self):
        inputs = keras.layers.Input(shape=(self.n_features,), name='observations')
        
        layer = keras.layers.Dense(
            units=10,
            activation='tanh',
            kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.3),
            bias_initializer=keras.initializers.Constant(0.1),  # Fixed bias initializer
            name='fc1'
        )(inputs)

        all_act = keras.layers.Dense(
            units=self.n_actions,
            activation=None,
            kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.3),
            bias_initializer=keras.initializers.Constant(0.1),  # Fixed bias initializer
            name='fc2'
        )(layer)

        self.all_act_prob = keras.layers.Softmax(name='act_prob')(all_act)
        self.model = keras.Model(inputs=inputs, outputs=self.all_act_prob)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

    @tf.function
    def _train_step(self, observations, actions, advantages):
        # Reshape observations if needed
        observations = observations.reshape(-1, self.n_features)
        
        with tf.GradientTape() as tape:
            logits = self.model(observations, training=True)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            loss = tf.reduce_mean(neg_log_prob * advantages)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def choose_action(self, observation):
        # Reshape observation to match expected input shape (None, n_features)
        observation = observation.reshape(-1, self.n_features)
        
        # Get action probabilities from model
        prob_weights = self.model(observation, training=False)
        
        # Choose action based on probabilities
        action = np.random.choice(range(self.n_actions), p=prob_weights.numpy().ravel())
        return action
    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def store_ob(self, s):
        self.ep_obs.append(s)

    def store_action(self, a):
        self.ep_as.append(a)

    def store_adv(self, r):
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs = self._discount_and_norm_rewards()
        loss = self._train_step(
            tf.convert_to_tensor(np.vstack(self.ep_obs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(self.ep_as), dtype=tf.int32),
            tf.convert_to_tensor(discounted_ep_rs, dtype=tf.float32)
        )
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return loss

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_data(self, pg_resume):
        self.model.save_weights(pg_resume + '.h5')

    def load_data(self, pg_resume):
        try:
            self.model.load_weights(pg_resume + '.h5')
        except:
            print(f"Could not load weights from {pg_resume}. Starting with fresh model.")