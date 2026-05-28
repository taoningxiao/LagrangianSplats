class Poly6:
	@staticmethod
	def hessian(r):
		# Given the normalized distance r = ||x|| / h
		# compute the hessian given by H = c1 * xx^T - c2 * I
		r = r.clamp(0, 1)
		t = (1 - r**2)
		c1 = 24 * t
		c2 = 6 * t**2
		return c1, c2

class WendlandC2:
	@staticmethod
	def original(r):
		# Given the normalized distance r = ||x|| / h
		r = r.clamp(0, 1)
		return (1 - r)**4 * (4 * r + 1)
	
	@staticmethod
	def gradient(r):
		# Given the normalized distance r = ||x|| / h
		# compute the gradient given by g = c * x
		r = r.clamp(0, 1)
		return (1 - r)**3 * (-20)

class WendlandC4:
	@staticmethod
	def gradient(r):
		# Given the normalized distance r = ||x|| / h
		# compute the gradient given by g = c * x
		r = r.clamp(0, 1)
		return (1 - r)**5 * (1 + 5 * r) * (-56)
	
	@staticmethod
	def laplacian(r, d):
		# Given the normalized distance r = ||x|| / h
		r = r.clamp(0, 1)
		return (1 - r)**4 * ((5. * d + 30) * r**2 - 4. * d * r - d)

	@staticmethod
	def hessian(r):
		# Given the normalized distance r = ||x|| / h
		# compute the hessian given by H = c1 * xx^T - c2 * I
		r = r.clamp(0, 1)
		t = (1 - r)**4
		c1 = 30 * t
		c2 = (1 + 4 * r - 5 * r**2) * t
		return c1, c2