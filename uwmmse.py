import pdb
import tensorflow as tf
import tensorflow_addons as tfa
from activations import *

# UWMMSE
class UWMMSE(object):
        # Initialize
        def __init__( self, Pmax=1., tx_antennas=1, rx_antennas=1, signal_dim=1, var=7e-10, feature_dim=1, batch_size=64, layers=4, learning_rate=1e-3, max_gradient_norm=5.0, exp='uwmmse' ):
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.variance          = var
            self.feature_dim       = feature_dim
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.exp               = exp
            self.global_step       = tf.Variable(0, trainable=False, name='global_step')
            self.T                 = tx_antennas
            self.R                 = rx_antennas
            self.d                 = signal_dim
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()
            
        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.complex128, shape=[None, None, None, self.R, self.T], name="H")
        
        # Normlize input to gcn
        def inp_norm(self):
            mu = tf.reduce_mean( self.H_gcn, axis=[-1], keepdims=True )
            sig = tf.cast( tf.math.reduce_std( self.H_gcn, axis=[-1], keepdims=True ), tf.complex128 )
            
            H = tf.math.divide( ( self.H_gcn - mu ), sig )
            return(H)
            
        # Declare complex trainable variables (omega)
        def dec_com_var(self, use_bias=True, name='_', shape=(1,1)):
            w_r = tf.compat.v1.get_variable( name=name+'w_r', shape=shape, initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
            w_i = tf.compat.v1.get_variable( name=name+'w_i', shape=shape, initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
            w = tf.complex(w_r,w_i)
            if use_bias:
                b_r = tf.compat.v1.get_variable( name=name+'b_r', shape=(shape[-1],), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                b_i = tf.compat.v1.get_variable( name=name+'b_i', shape=(shape[-1],), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                b = tf.complex(b_r,b_i)
                return(w,b)
            return(w)

        # Building network
        def build_network(self):
            # Retrieve number of nodes for initializing V
            self.nNodes = tf.shape( self.H )[1]
            
            # Extract diagonals
            self.dia = tf.transpose( tf.compat.v1.matrix_diag_part(tf.transpose(self.H,(0,3,4,1,2))), (0,3,1,2) )
            
            # Variance tensor
            self.var = tf.cast( self.variance * tf.eye( self.R, batch_shape=[self.nNodes], dtype=tf.float64 ), tf.complex128 )
            self.var = tf.tile( tf.expand_dims( self.var, axis=0 ), tf.constant( [self.batch_size,1,1,1] ) )

            if self.exp == 'uwmmse':
                # Learn GCN i/p 
                w, b = self.dec_com_var(name='1x1_conv', shape=(self.R * self.T,1))
                if self.T < self.R:
                    self.H_gcn = tf.reshape( tf.transpose( self.H, (0,1,2,4,3) ), [ self.batch_size, self.nNodes, self.nNodes, self.R * self.T ] )
                else:
                    self.H_gcn = tf.reshape( self.H, [ self.batch_size, self.nNodes, self.nNodes, self.R * self.T ] )
                self.H_gcn = tf.add( tf.matmul( self.H_gcn, w ), b ) 
                self.H_gcn = tf.squeeze(tf.squeeze( self.H_gcn ))
                self.H_gcn = self.inp_norm()
                
                # Diagonal for GCN i/p
                self.dH_gcn = tf.linalg.diag_part( self.H_gcn ) 
                self.dH_gcn = tf.compat.v1.matrix_diag( self.dH_gcn )
            
            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax/(2 * self.T * self.d))
            
            # Initialize V
            V = Vmax * tf.ones([self.batch_size, self.nNodes, self.T, self.d], dtype=tf.float64)
            V = tf.complex(V,V)
            self.V = V
            
            # Iterate over layers k
            for l in range(self.layers):
                with tf.compat.v1.variable_scope('Layer{}'.format(1), reuse=tf.compat.v1.AUTO_REUSE):
                    # Compute U^k
                    U = self.U_block( V )
                    self.U = U

                    # Compute W_wmmse^k
                    W_wmmse = self.W_block( U, V )
                    
                    if self.exp == 'wmmse':
                        # Compute V^k
                        V = self.V_block( U, W_wmmse, tf.complex(tf.cast(0.0, tf.float64),tf.cast(0.0, tf.float64)) )
                    else:
                        # Learn xi^k
                        a = self.gcn('a')

                        # Expand xi^k across MLP layers
                        aw1 = tf.expand_dims(a[:,:,:5],-2)
                        ab1 = tf.expand_dims(a[:,:,5:10],-2)
                        aw2 = tf.expand_dims(a[:,:,10:15],-1)
                        ab2 = tf.expand_dims(a[:,:,15:],-1)
                        
                        # Phi_xi^k
                        phi = cart_relu( tf.add( tf.matmul( cart_leaky_relu( tf.add( tf.matmul(W_wmmse, aw1), ab1 ) ), aw2 ), ab2 ) )
                           
                        # Compute W^k = W_wmmse + f_lin(W_wmmse;theta) 
                        W = W_wmmse + phi  
                        
                        # mu
                        mu = self.dec_com_var(use_bias=False, name='mu',shape=[1])

                        # Compute V^k
                        V = self.V_block( U, W, mu )
                    
                    ## Saturation non-linearity  ->  V if Tr(V'V) < Pmax ; V * sqrt(P)/Vfrobenius if Tr(V'V) > Pmax
                    norm = tf.math.real(tf.linalg.norm(V, ord='fro',axis=[-2,-1])) 
                    mask = tf.math.divide( tf.math.multiply(norm, ( 0.5 + 0.5 * tf.math.sign( tf.math.square(norm) - self.Pmax ) ) ), tf.math.sqrt(self.Pmax) ) + ( 0.5 + 0.5 * tf.math.sign( self.Pmax - tf.math.square(norm) ) )
                    mask = tf.cast( tf.expand_dims( tf.expand_dims(mask,axis=-1), axis=-1 ), tf.complex128 )

                    V = tf.math.divide( V, mask )
                    self.V = V
                    
            # Final V
            self.pow_alloc = tf.math.real(tf.linalg.norm(V, ord='fro',axis=[-2,-1]))
        
        def U_block(self, V):
            # Identity
            I = tf.cast(tf.eye(self.R, dtype=tf.float64),tf.complex128)

            # H_ii'V_i
            num = tf.compat.v1.matmul( self.dia, V )
   
            # sigma^2*I + sum_j( (H_ji V_jV_j' H_ji' )
            den = tf.reduce_sum( tf.matmul( self.H, tf.matmul( tf.expand_dims( tf.matmul( V, tf.linalg.adjoint( V ) ), axis=1 ), tf.linalg.adjoint( self.H ) ) ), axis=2 ) + self.var
                        
            # U = den^-1 num
            return( tf.compat.v1.matmul( tf.compat.v1.matrix_inverse( den + 1e-4*I ), num ) )

        # Sum-rate = z
        def W_block(self, U, V):
            # Identity
            I = tf.cast(tf.eye( self.d, batch_shape=[self.nNodes], dtype=tf.float64 ),tf.complex128)
            I = tf.tile( tf.expand_dims( I, axis=0 ), tf.constant( [self.batch_size,1,1,1] ) )
            
            # 1 - U_i' H_ii V_i
            den = I - tf.matmul( tf.linalg.adjoint( U ), tf.matmul( self.dia, V ) )
            #pdb.set_trace()
                        
            # W = den^-1
            return( tf.compat.v1.matrix_inverse( den ) )
       
        def V_block(self, U, W, mu):
            # Identity
            I = tf.cast(tf.eye(self.T, dtype=tf.float64),tf.complex128)
            
            # H_ii U_i W_i
            num = tf.matmul( tf.linalg.adjoint(self.dia), tf.matmul( U, W ) )
            
            # expand mu
            mu = tf.reshape( mu, shape=[1,1,1,1,1] )

            # sum_j( (H_ij' U_j W _j U_j' H_ij )
            den = tf.reduce_sum( tf.math.add( tf.matmul( tf.linalg.adjoint( self.H ), tf.matmul( tf.expand_dims( tf.matmul( tf.matmul( U, W ), tf.linalg.adjoint( U ) ), axis=2 ), self.H ) ), mu ), axis=1 )
                        
            # V = den^-1 num
            return( tf.compat.v1.matmul( tf.compat.v1.matrix_inverse( den + 1e-4*I ), num ) )        

        def gcn(self, name):
            # 2 Layers
            L = 2
            
            # Dims
            inp_dim = self.R+self.T
            input_dim = [inp_dim,32]
            output_dim = [32,16]        

            ## NSI [Batch_size X Nodes X Features]
            x = tf.concat([self.U,self.V], -2) #GV: the NSI is now the most recent U and V values at each node
            
            with tf.compat.v1.variable_scope('gcn_'+name):
                ##Reduce the NSI x
                w, b = self.dec_com_var(name='nsi', shape=(self.d, 1))
                x = tf.squeeze( cart_leaky_relu(tf.add(tf.matmul(x, w),b) ) )
    
                for l in range(L):
                    with tf.compat.v1.variable_scope('gc_l{}'.format(l+1)):
                        ## Weights, biases
                        w0, b0 = self.dec_com_var(name='tr0', shape=(input_dim[l], output_dim[l]))
                        w1, b1 = self.dec_com_var(name='tr1', shape=(input_dim[l], output_dim[l]))
                        
                        # XW
                        x1 = tf.matmul(x, w1)
                        x0 = tf.matmul(x, w0)
                        
                        # diag(A)XW0 + AXW1
                        x1 = tf.matmul(self.H_gcn+.001, x1)  
                        x0 = tf.matmul(self.dH_gcn, x0)
                        
                        ## AXW + B
                        x1 = tf.add(x1, b1)
                        x0 = tf.add(x0, b0)
                        
                        # Combine
                        x = x1 + x0
                        
                        # activation(AXW + B)
                        if l == L-1:
                            x = cart_leaky_relu(x)  
                        else:
                            x = cart_leaky_relu(x)

                # Output
                    output = x
            
            return output
        
        def build_objective(self):                        
            # H_ii V_i V_i' H_ii'
            num = tf.matmul( tf.matmul( tf.matmul( self.dia, self.V ), tf.linalg.adjoint( self.V ) ), tf.linalg.adjoint( self.dia ) )

            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
            den = tf.reduce_sum( tf.matmul( self.H, tf.matmul( tf.expand_dims( tf.matmul( self.V, tf.linalg.adjoint( self.V ) ), axis=1 ), tf.linalg.adjoint( self.H ) ) ), axis=2 ) + self.var - num 

            # rate = log(1 + SINR)
            self.rate = tf.math.log( tf.math.real( tf.compat.v1.matrix_determinant( tf.cast( tf.eye( self.R, batch_shape=[self.nNodes], dtype=tf.float64 ), tf.complex128 ) + tf.matmul( tf.compat.v1.matrix_inverse( den ), num ) ) ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
            
            # Sum Rate = sum_i ( rate )
            self.utility = tf.reduce_sum( self.rate, axis=1 )
            self.util = tf.reduce_mean( self.rate, axis=1 )
            
            # Minimization objective
            self.obj = -tf.reduce_mean( self.util )
            
            if self.exp == 'uwmmse':
                self.init_optimizer()

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()

            # Optimizer
            self.opt = tfa.optimizers.NovoGrad(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            self.gradients = tf.gradients(self.obj, self.trainable_params)
            
            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)

            # Update the model
            self.updates = self.opt.apply_gradients(zip(clip_gradients, self.trainable_params))#, global_step=self.global_step)
                
        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs, inp_gcn=None):
            input_feed = dict()
            input_feed[self.H.name] = inputs

            output_feed = [self.obj, self.utility, self.pow_alloc, self.updates]
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]

        def eval(self, sess, inputs, inp_gcn=None):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            
            output_feed = [self.obj, self.utility, self.pow_alloc]
            outputs = sess.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]
