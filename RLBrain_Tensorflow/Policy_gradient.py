import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

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
            # TensorBoard logging
            self.log_dir = "logs/"
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def _build_net(self):
        # Define inputs using tf.keras.Input
        inputs = tf.keras.Input(shape=(self.n_features,), name="observations")
        self.tf_obs = inputs

        # Define hidden layer (fc1)
        layer = layers.Dense(
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
        )(inputs)

        # Define output layer (fc2)
        all_act = layers.Dense(
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        # Use softmax to calculate probabilities
        all_act_prob = layers.Softmax(name='act_prob')(all_act)
        self.all_act_prob = all_act_prob

        # Define the model
        self.model = tf.keras.Model(inputs=inputs, outputs=all_act_prob)

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=self.lr)


    def choose_action(self, observation):
        #observation = np.expand_dims(observation, axis=0)  # Add batch dimension
        prob_weights = self.model.predict(observation)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # Select action based on probabilities
        return action

    def store_ob(self, s):
        self.ep_obs.append(s)

    def store_action(self, a):
        self.ep_as.append(a)

    def store_adv(self, r):
        self.ep_rs.append(r)

    def learn(self, all_ob, all_action, all_adv):
        # Discount and normalize rewards
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # Convert episode data to arrays
        # all_ob = np.vstack(self.ep_obs)
        # all_action = np.array(self.ep_as)
        # all_adv = discounted_ep_rs_norm

        with tf.GradientTape() as tape:
            logits = self.model(all_ob, training=True)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=all_action)
            loss = tf.reduce_mean(neg_log_prob * all_adv)  # Reward-guided loss

        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Clear episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return loss.numpy()

    def _discount_and_norm_rewards(self):
        # Compute discounted rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # Normalize rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_data(self, path):
        self.model.save_weights(path)

    def load_data(self, path):
        self.model.load_weights(path)
