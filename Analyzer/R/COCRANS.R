install.packages('nonpar')
library('nonpar')

data <- cbind(c(314,65,41),c(67,311,42),c(24,24,372))
cochrans.q(data)