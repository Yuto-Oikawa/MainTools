data <- cbind(c(147,34,7,167,33,17),c(38,154,11,27,157,13),c(25,22,192,16,20,180))
rownames(data) <- c("before実際1次","before実際1.5次","before実際2次","after実際1次","after実際1.5次","after実際2次")
colnames(data) <- c("予測1次","予測1.5次","予測2次")

# before=差し替え前が1, after=差し替え後が2
grp <- as.factor(c(1,1,1,2,2,2))

install.packages("vegan")
library("vegan")

anosim(data,grouping=grp)
#plot(result)