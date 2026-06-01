# Install once
install.packages("quantmod")

library(quantmod)

# Download Bitcoin daily data from Yahoo Finance
getSymbols("BTC-USD", src = "yahoo",
           from = "2015-01-01",
           to = Sys.Date())

btc <- `BTC-USD`

# Convert to data frame
btc_df <- data.frame(
  Date = index(btc),
  coredata(btc)
)

# Save as CSV
write.csv(btc_df, "BTC_USD_daily.csv", row.names = FALSE)

# View first rows
head(btc_df)

getwd()