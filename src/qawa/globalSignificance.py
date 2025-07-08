import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
#from tensorflow.keras.utils import register_keras_serializable

#@register_keras_serializable()

class globalSignificance(tf.keras.metrics.Metric):
	
	def init(self, name='globalSig',threshold = 0.5,sigSumWeight = None,bkgSumWeight = None, **kwargs):
		super(globalSignificance, self).init(name=name, **kwargs)
		self.sigPart = self.add_weight(name='sigPart', initializer='zeros')
		self.bkgPart = self.add_weight(name='bkgPart', initializer='zeros')
		self.sigPartTotal = self.add_weight(name='sigPartTotal', initializer='zeros')
		self.bkgPartTotal = self.add_weight(name='bkgPartTotal', initializer='zeros')
		self.threshold = threshold
		self.sigSumWeight = sigSumWeight.astype('float32')
		self.bkgSumWeight = bkgSumWeight.astype('float32')
	

	def update_state(self, y_true, y_pred, sample_weight=None):
		y_true = tf.cast(y_true, tf.bool) 
		y_pred = tf.squeeze(y_pred)
		y_pred2 = tf.cast(tf.greater(y_pred,tf.constant(0.0)), tf.bool)
		y_pred = tf.cast(tf.greater(y_pred,self.threshold), tf.bool)
		isSig = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
		isSig = tf.cast(isSig, self.dtype)
		isBkg = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
		isBkg = tf.cast(isBkg, self.dtype) 
		isTotalSig = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred2, True))
		isTotalSig = tf.cast(isTotalSig, self.dtype) 
		isTotalBkg = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred2, True))
		isTotalBkg = tf.cast(isTotalBkg, self.dtype) 
		if sample_weight is not None:
			sample_weight = tf.cast(sample_weight, self.dtype)
			isSig = tf.multiply(isSig, sample_weight) 
			isBkg = tf.multiply(isBkg, sample_weight) 
			isTotalSig = tf.multiply(isTotalSig, sample_weight) 
			isTotalBkg = tf.multiply(isTotalBkg, sample_weight) 
		self.sigPart.assign_add(self.sigSumWeighttf.reduce_sum(isSig))
		self.bkgPart.assign_add(self.bkgSumWeighttf.reduce_sum(isBkg))
		self.sigPartTotal.assign_add(self.sigSumWeighttf.reduce_sum(isTotalSig))
		self.bkgPartTotal.assign_add(self.bkgSumWeighttf.reduce_sum(isTotalBkg))
	
	def result(self):
		return 1.513274*tf.math.divide(tf.math.multiply(tf.math.sqrt(self.sigSumWeight+self.bkgSumWeight),self.sigPart),\
                          tf.math.multiply(tf.sqrt(self.bkgPartTotal + self.sigPartTotal),tf.math.sqrt(self.bkgPart)))
	def reset_state(self):
		self.sigPart.assign(0)
		self.bkgPart.assign(0)
		self.sigPartTotal.assign(0)
		self.bkgPartTotal.assign(0)