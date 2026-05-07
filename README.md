**ARECANUT PRICE PREIDCTION USING WEATHER DATA**

Arecanut is among the important plantation crops in the coastal region of Karnataka from an 
economic point of view, but predicting prices for this commodity has been a challenge. Seasonal 
and monsoon-driven price patterns are observed after a time lag, along with significant 
inter-market variations, leaving farmers only with past experience in making selling decisions. 
This study attempts to address the problem directly. Using 26 years of weekly data spanning 
January 2000 to December 2025, three datasets were constructed and integrated: wholesale 
Arecanut prices from five major APMC markets across coastal Karnataka (Mangaluru, Puttur, 
Sagar, Shimoga, and Sirsi), ERA5-Land weekly temperature records for seven locations across 
Dakshina Kannada and Udupi, and IMD RF25 gridded weekly rainfall data for the same region.

A Linear Regression model was built using price lag features combined with temperature and 
rainfall variables. The price lag features (lag 1 and lag 2 prices along with a linear time index) 
capture the autoregressive price trend, while the weather features (weekly mean temperature, 
mean rainfall, their standard deviations, one-week lagged temperature, and rainfall lag features at 
4, 8, 12, and 16 weeks derived from the Pearson correlation analysis) bring in the climatic 
dimension. A chronological 80/20 train-test split was applied, with 785 weeks for training and 
193 weeks for testing (March 2022 to December 2025), with time ordering enforced to prevent 
data leakage. 

Alongside the forecasting model, a Pearson lag-correlation analysis was conducted between 
monsoon-period rainfall and Arecanut prices across lag intervals of 1 to 24 weeks. This 
confirmed a consistently negative lagged relationship between rainfall and Arecanut prices 
during June-September, with the strongest impact at lags of 4-16 weeks, consistent with the 
crop's post-harvest supply cycle. 

The main finding is that January and February are the months with the best prices to sell 
Arecanut in 88.5% of the years in the sample, while September is the worst month in 96.2% of 
the years. While the differential between the best and worst month was between Rs 1,000 and Rs 
2,500 per quintal in the early 2000s, it has shot up to between Rs 9,000 and Rs 11,000 per quintal 
in recent years.
