
def DisCo(y_true, y_pred,alpha = 1000):
    from tensorflow.keras.losses import binary_crossentropy, CategoricalCrossentropy,sparse_categorical_crossentropy
    from tensorflow.keras import backend as K
    import tensorflow as tf
    #alpha determines the amount of decorrelation; 0 means no decorrelation.
    #Note that the decorrelating feature is also used for learning.
    
    domain_mask = tf.where(y_true<10,K.ones_like(y_true),K.zeros_like(y_true))
    
    X = tf.boolean_mask(y_pred, tf.squeeze(domain_mask))
    Y = tf.boolean_mask(y_true, tf.squeeze(domain_mask))

    loss_class = sparse_categorical_crossentropy(Y,X)

    
    X = y_pred
    Y = tf.cast(domain_mask,dtype=tf.float32)    
    LY = K.shape(Y)[0]

    Y=K.reshape(Y,shape=(LY,1))    
    
    r = K.sum(X*X, axis=1)
    r = K.reshape(r, [-1, 1])
    ajk = r - 2*K.dot(X, K.transpose(X)) + K.transpose(r)
    
    bjk = K.square(K.reshape(K.repeat(Y,LY),shape=(LY,LY)) - K.transpose(Y))
    

    Ajk = ajk - K.mean(ajk,axis=0)[None, :] - K.mean(ajk,axis=1)[:, None] + K.mean(ajk)
    Bjk = bjk - K.mean(bjk,axis=0)[None, :] - K.mean(bjk,axis=1)[:, None] + K.mean(bjk)
    dcor = K.sum(Ajk*Bjk) / K.sqrt(K.sum(Ajk*Ajk)*K.sum(Bjk*Bjk))    
    return loss_class + alpha*dcor
