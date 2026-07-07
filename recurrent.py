import tensorflow as tf

class LSTMSNPCell(tf.keras.layers.Layer):
    # 1.类初始化
    def __init__(self, units, use_compensation=False, lambda_c=0.5):
        super(LSTMSNPCell, self).__init__()
        self.units = units # 神经元数量（隐藏状态的维度）
        self.use_compensation = use_compensation # 是否使用补偿状态
        self.lambda_c = lambda_c # 0.5初始

        self.state_size = [units, units] # h,u
        self.output_size = units
        # junfu：===== dynamics record =====
        self.state_history = [] # 用于保存神经元历史状态
        self.comp_history = []

    def build(self, input_shape): 
    # 2.如果启用补偿状态5组门控(r,c,o,g,a)，否则只有4组门控(r,c,o,a)
        input_dim = input_shape[-1]
        gate_num = 5 if self.use_compensation else 4
        
        # 作用于当前输入input_dim，
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * gate_num),
            initializer='glorot_uniform',
            name='kernel'
        )
        # 作用于上一时刻隐藏状态h(t-1)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * gate_num),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        # 偏置
        self.bias = self.add_weight(
            shape=(self.units * gate_num,),
            initializer='zeros',
            name='bias'
        )

    def call(self, inputs, states):
        # 3.前向计算
        h_tm1, u_tm1 = states # states是上一时刻的h(t-1)与u(t-1)

        z_all = ( # 一次性计算出的所有门控预激活值，然后按维度 units 切分成 r,c,o,a,g
            tf.matmul(inputs, self.kernel)
            + tf.matmul(h_tm1, self.recurrent_kernel)
            + self.bias
        )

        if self.use_compensation:
            r, c, o, a, g = tf.split(z_all, 5, axis=1)
        else:
            r, c, o, a = tf.split(z_all, 4, axis=1)

        r = tf.nn.sigmoid(r) # 值域[0,1]
        c = tf.nn.sigmoid(c)
        o = tf.nn.sigmoid(o)
        a = tf.nn.tanh(a) # 值域[-1,1]

        # ===== no compensation (4门控的时候)=====
        if not self.use_compensation: #  公式： u_t = r⊙u_{t-1}-c⊙a  
                                            #  h_t = o⊙tanh(u_t)
            u = r * u_tm1 - c * a

            # 保存状态演化
            self.last_state_norm = tf.reduce_mean(tf.abs(u))

        
        # ===== compensation （5门控的时候）=====
        else: # 公式：comp = λ_c·g⊙h_{t-1}
                #    u_t = r⊙u_{t-1} - c ⊙ a  +  comp
                #    h_t = o ⊙ tanh(u_t)
            
            g = tf.nn.sigmoid(g) # 补偿门采用Sigmoid函数激活，用于动态调节补偿强度

            comp = self.lambda_c * g * h_tm1

            u = r * u_tm1 - c * a + comp

            # ===== 保存状态演化 =====
            self.last_state_norm = tf.reduce_mean(tf.abs(u))

            # ===== 保存补偿强度 =====
            self.last_comp_norm = tf.reduce_mean(tf.abs(comp))

        h = o * tf.nn.tanh(u)

        return h, [h, u]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype)
        ]