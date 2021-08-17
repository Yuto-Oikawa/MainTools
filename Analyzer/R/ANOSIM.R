data <- cbind(c(147,34,7,167,33,17),c(38,154,11,27,157,13),c(25,22,192,16,20,180))
rownames(data) <- c("beforeÀÛ1Ÿ","beforeÀÛ1.5Ÿ","beforeÀÛ2Ÿ","afterÀÛ1Ÿ","afterÀÛ1.5Ÿ","afterÀÛ2Ÿ")
colnames(data) <- c("—\‘ª1Ÿ","—\‘ª1.5Ÿ","—\‘ª2Ÿ")

# before=·‚µ‘Ö‚¦‘O‚ª1, after=·‚µ‘Ö‚¦Œã‚ª2
grp <- as.factor(c(1,1,1,2,2,2))

install.packages("vegan")
library("vegan")

anosim(data,grouping=grp)
#plot(result)