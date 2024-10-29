#### SER494: Project Proposal

# Clustering Analysis of Steam Games Using Review Sentiment and Game Features

#### Anonymous

#### 10/28/2024

  

---

  

## Keywords:

Game Review Sentiment Analysis, Steam Game Clustering, Consumer Patterns in Gaming

  

---

  

1. Game Review Sentiment Analysis
2. Steam Game Clustering
3. Consumer Patterns in Gaming


  
## Description: 
This project centers on clustering Steam games using a range of attributes, including user review sentiment, genre, category, and more, to uncover distinct patterns that reflect consumer preferences. By analyzing these factors, the study aims to identify clusters of games that can guide developers, marketers, and consumers alike. This approach may help consumers find games that align with their interests across a range of factors they may not have initially considered, such as genre, price, category, or sentiment trends in user reviews. The clusters generated from this analysis could serve as a useful tool for navigating the expansive game library on Steam, helping players discover titles that align closely with their preferences and play style. It may also offer valuable information for developers and marketers who seek to understand and respond to user feedback more effectively.
  


## Research Questions (ROs):
- RO1: To describe the trends within the Steam game data, focusing on features such as review sentiment, genre, category, and pricing.
- RO2: To cluster games based on their review sentiment, genre, and other attributes to identify distinct groupings that reflect various consumer patterns and preferences.
- RO3: To evaluate the defensibility of the clustering model used in RO2, ensuring its reliability in accurately grouping games based on user preferences.
- RO4: To analyze and interpret the relationships within the clustered groups to understand how factors like genre and review sentiment correlate with consumer engagement and satisfaction.



## Intellectual Merit: 
The potential of this project is in advancing knowledge around user behavior and game characteristics in the digital gaming market, particularly through novel combinations of review sentiment analysis and clustering of game features. By leveraging diverse game attributes—such as genre, category, pricing, and review sentiment—this project will explore nuanced consumer patterns within the gaming ecosystem. The clustering outcomes have the potential to uncover previously unobserved correlations between game attributes and user satisfaction, helping to identify what defines different types of games from a consumer’s perspective. This research could reveal new insights into how specific game features influence user preferences, and ultimately, how they drive engagement and satisfaction in different gaming niches. These insights can contribute meaningfully to fields like digital marketing, game development, and consumer behavior analysis, offering data-driven strategies for better understanding and meeting the needs of diverse gaming audiences.

  
  

## Data Sourcing: 
This project primarily uses data sourced from the Steam Store, a platform that provides detailed game information, including game titles, genres, pricing, user reviews, and more. The Steam Store data offers a foundation for understanding the primary attributes of each game, which are essential for clustering analysis. To incorporate review sentiment analysis, this project will retrieve individual game review data using the Steam Web API, which allows access to publicly available user reviews and metadata. It may also use a python package. For the sentiment analysis of review text, the data will be preprocessed using a Python package such as TextBlob or VADER, which are widely used for analyzing sentiment in natural language text.
  
  

## Background Knowledge: 
###  How Price Sensitivity Affects Consumer Buying Behavior:
- This article explores how price sensitivity and promotional pricing influence consumer decisions in e-commerce, providing knowledge into the effectiveness of discount strategies in driving impulse buys and long-term customer retention.
- Link: https://stores-goods.com/blog/consumer-behavior/how-price-sensitivity-affects-consumer-buying/

### How Price Sensitivity Analysis Can Reveal Consumer Behavior:
- This source discusses the importance of understanding price sensitivity to optimize pricing strategies and drive sales. It covers how consumers react to price changes and why price elasticity is crucial for maximizing revenue.
- Link: https://borderlessaccess.com/blog/understanding-how-customer-behavior-is-shaped-by-price-sensitivity/

### The Psychology of Discounts: 8 Researched-Backed Strategies:
- This article delves into the psychological tactics behind discounts, such as urgency and scarcity, and how they motivate consumers to act quickly. It emphasizes the importance of simple pricing and the influence of ending prices in .99.
- Link: https://www.namogoo.com/blog/consumer-behavior-psychology/psychology-of-discounts/

## Related Work: 

### A meta-analysis of the impact of price presentation on perceived savings

DOI: https://doi.org/10.1016/S0022-4359(02)00072-6
- Many findings, here are a few:
	- Consumers value savings on bundles less as the bundle size increases. This finding is open to multiple hypotheses as to the cause and would be fruitful area for future research.
	- Deals are more effective if they are less consistent (predictable) and more distinctive.
	- Including free gifts in general lowers the perceived value of the deal.
#### Citation

Aradhna Krishna, Richard Briesch, Donald R. Lehmann, Hong Yuan,
A meta-analysis of the impact of price presentation on perceived savings,
Journal of Retailing,
Volume 78, Issue 2,
2002,
Pages 101-118,
ISSN 0022-4359,
https://doi.org/10.1016/S0022-4359(02)00072-6.
(https://www.sciencedirect.com/science/article/pii/S0022435902000726)
Keywords: Meta analysis; Behavioral pricing; Reference price; Pricing; Promotions; Consumer behavior


### Consumers' Impulsive Buying Behavior in Online Shopping Based on the Influence of Social Presence

DOI: https://doi.org/10.1155/2022/6794729
- The concept of social presence comes from the field of psychology. It has attracted attention from brand management, marketing, and communication studies. Second, the academic contributions of this study mainly include the following two aspects: the variables affect the social presence of platforms and establishing the relevant of impulsive buying behaviors. 

#### Citation

Zhang, Mingming, Shi, Guicheng, Consumers’ Impulsive Buying Behavior in Online Shopping Based on the Influence of Social Presence, _Computational Intelligence and Neuroscience_, 2022, 6794729, 8 pages, 2022. [https://doi.org/10.1155/2022/6794729](https://doi.org/10.1155/2022/6794729)