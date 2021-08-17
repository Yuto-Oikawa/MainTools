pred1 <- c(0.78,0.70,0.74)
pred1.5 <- c(0.76,0.73,0.75)
pred2 <- c(0.80,0.91,0.86)
after.pred1 <- c(0.77,0.80,0.78)
after.pred1.5 <- c(0.80,0.75,0.77)
after.pred2 <- c(0.83,0.86,0.85)

Replace <- rep(c(1, 2), c(3, 3))
Replace <- factor(Replace, levels=c(1, 2), labels=c("·‚µ‘Ö‚¦‘O", "·‚µ‘Ö‚¦Œã"))

result <- manova(rbind(pred1, pred1.5, pred2, after.pred1, after.pred1.5, after.pred2) ~ Replace)
summary(result, test="Wilks")