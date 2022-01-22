import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(False)


class ActorCritic:
    def __init__(self,alpha,beta,gamma,state_dim,action_dim,num_of_layers=2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_of_layers
        self.actor_net, self.policy_net = self.build_policy_network(self.state_dim,self.action_dim,self.num_layers)
        self.critic_net = self.build_critic_network(self.state_dim,self.num_layers)
        self.action_space = [i for i in range(self.action_dim)]
        
    def build_policy_network(self,state_dim,action_dim,num_of_layers=2):
        state_input = tf.keras.Input(shape=(state_dim,))
        delta = tf.keras.Input(shape=[1])
        layer = tf.keras.layers.Dense(24,activation='relu')(state_input)
        for i in range(num_of_layers):
            layer = tf.keras.layers.Dense(24,activation='relu')(layer)
        output = tf.keras.layers.Dense(action_dim,name='output')(layer)
        softmax = tf.keras.layers.Softmax()(output)
        concatenate_layer = tf.keras.layers.Concatenate(axis=1,trainable=False)([softmax,delta])
        
        #def custom_policy_loss(y_true,y_pred,delta):
        #    out = tf.keras.backend.clip(y_pred,1e-8,1-1e-8)
        #    log_likhood = tf.keras.backend.batch_dot(y_true,out)

        #    return tf.keras.backend.sum(-log_likhood,delta)
        def custom_policy_loss(y_true,y_pred):
            #print(f"delat:{delta}")
            delta = y_pred[0][-1]
            #d = tf.keras.backend.get_value(delta)
            #print(f"d: {d}")
            y_pred = y_pred[0][:-1]
            out = tf.clip_by_value(y_pred,1e-8,1-1e-8)
            #print(f"out:{out}")
            log_likhood = y_true*tf.math.log(out)
            #print(f"ll:{log_likhood}")
            return tf.reduce_sum(-log_likhood*delta)
        policy = tf.keras.models.Model(inputs=state_input,outputs=[softmax])
        actor = tf.keras.models.Model(inputs=[state_input,delta],outputs=[concatenate_layer])
        #actor.add_loss(custom_policy_loss(state_input,output,delta))
        actor.compile(loss = custom_policy_loss,optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        
        return actor,policy
    
    def build_critic_network(self,state_dim,num_of_layers):
        layers = tf.keras.layers.Dense(16,activation='relu')
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16,input_shape=(state_dim,),name='input'))
        for i in range(num_of_layers):
            model.add(layers)
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=self.beta))
        
        return model

    def choose_choose_action(self,state):
        state = state[np.newaxis,:]
        prob = self.policy_net.predict(state)[0]
        action = np.random.choice(self.action_space ,p=prob)

        return action

    def learn(self,state,action,reward,next_state,done):
        state = state[np.newaxis,:]
        next_state = next_state[np.newaxis,:]
        
        current_state_critic_value = self.critic_net.predict(state)
        next_state_critic_value = self.critic_net.predict(next_state)
        
        target = reward + (self.gamma * next_state_critic_value * (1-int(done)))
        delta = target - current_state_critic_value
        
        actions = np.zeros([1,self.action_dim])
        actions[np.arange(1),action] = 1.0
        
        self.actor_net.fit(x=[state,delta],y=actions,verbose=0)
        self.critic_net.fit(x=state,y=target,verbose=0)