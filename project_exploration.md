#### SER494: Exploratory Data Munging and Visualization
#### Exploring Review Trends and Sale Period Dynamics in Steam Games
#### Detim Zhao
#### 10/20/2024

## Basic Questions
Dataset Link: https://www.kaggle.com/datasets/amanbarthwal/steam-store-data?resource=download
**Dataset Author(s):** Aman Barthwal (is the owner of the dataset on Kaggle).

**Dataset Construction Date:** Date Scraped: 19th, May 2024. File was last modified May 25, 2024. 

**Dataset Record Count:** 42,497 

**Dataset Field Meanings:** Explanations below...
- `app_id`: Unique identifier for each game
- `title`: The name of the application, a product on the store, typically a game. 
  - This column will be renamed to `game_title` for clarity, as I did not see anything included in the dataset that is not a game, e.g. Valve Index Headset (a VR headset made by Valve you can buy on the Steam Store), which is VR hardware and not a game.  
- `release_date`: The date the game was released on Steam
- `genres`: The Steam game's genre
- `categories`: Features or tags of the game (e.g., multiplayer, single-player)
- `developer`: The studio or people that made/developed the game
- `publisher`: The company that published the game
- `original_price`: The price of the game without discounts. I will rename this column to `original_price_INR` for clarity since the dataset's prices are in rupees (INR).
- `discount_percentage`: The percentage reduction from the original price
  -  e.g., a discount of -75% means the game is being sold at 25% of its original price (this is likely due to how the Steam store displays the percentage off of a game to the user). 
  -  This column will be renamed to `percentage_of_original_price` to avoid confusion.
- `discounted_price`: The price after applying the discount (if applicable). 
  - I will rename this column to `discounted_price_INR` for clarity since the dataset's prices are in rupees (INR).
- `dlc_available`: Whether downloadable content (DLC) is available for the game
- `age_rating`: The age restriction for the game
  - I will rename this to `age_restricted` where true is that is restricted and false if it is not.
- `content_descriptor`: Describes specific elements of the game content, such as violence, language, or mature themes
  - This will be renamed to `content_descriptor_tags` to avoid any confusion.
- `win_support`: Indicates if the game has windows support (Boolean)
- `mac_support`: Indicates if the game has macOS support (Boolean)
- `linux_support`: Indicates if the game has Linux support (Boolean)
- `about_description`: A textual description of the game, often provided by the developer or publisher
  - This will be renamed to `store_game_description`, which is a bit verbose, but it clarifies the description is from the Store page of the game. 
- `awards`: Number of awards the game has received
  - This will be renamed to `award_count` for clarity
- `overall_review`: A qualitative summary of the overall user reviews (e.g., Mostly Positive)
- `overall_review_%`: The percentage of positive overall user reviews
  - Because this column indicates how positive the value of the review is (0 to 100), where lower is more negative while higher is more positive, I will rename this column to `overall_positive_review_percentage` to avoid initial confusion. 
- `overall_review_count`: The total number of user reviews
- `recent_review`: Similar to the overall_review (qualitative summary), but it is considering recent user reviews 
- `recent_review_%`: Similar to overall_review_%, but it considers recent reviews percentages
  - Again based on overall_review_%, and detailing how positive the value is, I will rename this column to `recent_positive_review_percentage`
- `recent_review_count`: Similar to overall_review_count, but it considers recent review count

**Dataset File Hash(es):** e59d2a2cfe2cf8c08bc31818016f97fd
- Tool used: https://emn178.github.io/online-tools/md5_checksum.html



## Interpretable Records
### Record 1
**Raw Data:** From csv:
2357570,Overwatch® 2,"10 Aug, 2023","Action, Free to Play","Online PvP, Online Co-op, Cross-Platform Multiplayer, Steam Achievements, In-App Purchases","Blizzard Entertainment, Inc.","Blizzard Entertainment, Inc.",,,Free,2,0,,"Overwatch 2 is a critically acclaimed, team-based shooter game set in an optimistic future with an evolving roster of heroes. Team up with friends and jump in today.",True,False,False,0,Overwhelmingly Negative,18.0,266951.0,Mostly Negative,34.0,8253.0


**Interpretation:** Overwatch 2 
- App ID: 2357570
- Title: Overwatch® 2
- Release Date: August 10, 2023
- Genres: Action, Free to Play
- Categories: Online PvP, Online Co-op, Cross-Platform Multiplayer, Steam Achievements, In-App Purchases
- Developer and Publisher: Blizzard Entertainment, Inc.
- Original Price: Not listed
- Discount Percentage: Not listed
- Discounted Price: Free
- DLC Available: 2 downloadable content (DLC) items are available for purchase.
- Age Rating: The game is not age-restricted (0 rating).
- Content Descriptor: Not listed.
- About Description: "Overwatch 2 is a critically acclaimed, team-based shooter game set in an optimistic future with an evolving roster of heroes. Team up with friends and jump in today."
- Platform Support: Available on Windows (no support for Mac or Linux).
- Awards: None listed.
- Overall Review: "Overwhelmingly Negative" with only 18% positive reviews out of 266,951 total reviews.
- Recent Review: "Mostly Negative" with 34% positive reviews from 8,253 recent reviews.

This record seems reasonable because "Overwatch 2" is a free-to-play, online multiplayer game developed and published by Blizzard Entertainment. Released in August 2023, it offers various multiplayer modes, cross-platform multiplayer, and in-app purchases. It is a sequel to Overwatch. The overwhelmingly negative reviews are due to community feedback on the game when it was added to Steam in 2023. At that time, it was one of the worst rated games on Steam according to game news outlets. The free-to-play aspect explains the lack of pricing data, and the review scores match the real-world criticism the game received. The absence of content descriptors and awards is expected, as it’s common for free-to-play games, and the DLC availability aligns with Blizzard's monetization strategy. 



### Record 2
**Raw Data:** From csv:
2341330,Lost Alone Ultimate,"25 Apr, 2023","Action, Adventure, Indie, Strategy","Single-player, Steam Achievements, Steam Trading Cards, Family Sharing",Daniele Doesn't Matter,Doesn't Matter Games,,,₹690.00,1,0,,"Lost Alone Ultimate is a first-person psychological horror game designed to convey anxiety, distress, and terror. You will have to explore three different houses, avoid staying in the dark, and face moments of fear that will not give you respite.",True,False,False,0,Very Positive,89.0,66.0,,,


**Interpretation:** Lost Alone Ultimate
- App ID: 2341330
- Title: Lost Alone Ultimate
- Release Date: April 25, 2023
- Genres: Action, Adventure, Indie, Strategy
- Categories: Single-player, Steam Achievements, Steam Trading Cards, Family Sharing
- Developer: Daniele Doesn't Matter
- Publisher: Doesn't Matter Games
- Original Price: Not listed
- Discount Percentage: Not listed
- Discounted Price: ₹690.00
- DLC Available: No downloadable content (DLC) available.
- Age Rating: The game is not age-restricted (0 rating).
- Content Descriptor: Not listed.
- About Description: "Lost Alone Ultimate is a first-person psychological horror game designed to convey anxiety, distress, and terror. You will have to explore three different houses, avoid staying in the dark, and face moments of fear that will not give you respite."
- Platform Support: Available on Windows (no support for Mac or Linux).
- Awards: None listed.
- Overall Review: "Very Positive" with 89% positive reviews out of 66 total reviews.
- Recent Review: Not available.

This record seems reasonable because "Lost Alone Ultimate" is a psychological horror game, released in April 2023 by indie developer Daniele Doesn't Matter and published by Doesn't Matter Games. Priced at ₹690.00, the game emphasizes exploration and survival within a horror setting. It supports single-player mode that has Steam Achievements and Steam Trading Cards but has no additional downloadable content (DLC). The game has received "Very Positive" overall feedback, with 89% of the 66 reviews being positive. However, there are no recent reviews available, and the game doesn't have an age rating or content descriptor. The absence of pricing data (original price) could be due to regional factors or discounts, but the discounted price is in rupees, which indicates the dataset’s regional focus. This means that the data scrapped was in the context of rupees in the Steam store. The "Very Positive" reviews would make sense as feedback from an indie horror game for a niche audience, and the lack of recent reviews suggests that it hasn't gained much recent attraction. 



## Background Domain Knowledge
### Introduction to the Domain: Consumer Behavior in Gaming and Pricing Strategies

In the modern age, the gaming industry has become one of the largest and most influential sectors in entertainment, with platforms like Steam acting as major marketplaces for game distribution. Understanding consumer behavior in this space is essential, particularly how pricing strategies and discount periods influence purchasing decisions. Pricing is not just a reflection of value but a powerful tool that shapes consumer perception and drives purchasing behavior. Key elements such as price sensitivity, discount tactics, and psychological influences all play a significant role in the way consumers engage with games on platforms like Steam.

**Price Sensitivity and Consumer Behavior**  
Price sensitivity refers to how consumers react to price changes, which in turn affects their buying behavior. In the gaming industry, where games vary greatly in price—from indie titles to AAA blockbusters—price sensitivity can determine a product's success. According to research on consumer behavior in e-commerce, price-sensitive customers are likely to be influenced by discount strategies, especially during sales periods. Sales events, such as Steam’s famous seasonal sales, create an environment where users often make purchasing decisions based on perceived savings, even if the purchase was not initially intended .

This is where **price elasticity** comes into play—measuring how changes in price influence demand. In digital marketplaces like Steam, price elasticity can vary. For example, consumers may be more willing to purchase indie games when prices drop significantly, while blockbuster titles might retain higher demand despite smaller discounts. Understanding the price elasticity for different game types can help developers and publishers optimize their pricing strategies .

**The Psychology of Discounts and Perceived Value**  
Discounts are a powerful motivator in driving consumer behavior, and their effectiveness goes beyond simple price reductions. Studies in consumer psychology show that the manner in which discounts are presented—such as urgency or scarcity—can significantly affect buying decisions . For instance, presenting discounts as "limited-time offers" or "exclusive deals" can trigger a fear of missing out (FOMO), leading consumers to make impulse purchases.

Furthermore, discounts create a psychological anchor, making the discounted price appear more favorable compared to the original. This effect is particularly strong during large-scale sales events like Steam’s summer sale, where the contrast between the original and discounted prices encourages purchases. Research also shows that consumers perceive discounts ending in .99 as being cheaper than whole numbers, which can drive a higher rate of conversion .

**Impact of Social Presence and Impulse Buying**  
Another aspect of consumer behavior in digital marketplaces is the influence of social presence on impulse buying. In online environments, particularly those with integrated social features such as Steam's community discussions and friend activity streams, consumers are more likely to make impulsive purchases when they perceive a social presence. Seeing what friends are playing or purchasing, or observing positive reviews from other users, can lead to impulse buys, especially during high-discount sales .

This phenomenon ties into broader trends in **impulse buying behavior**—where unplanned purchases are made due to emotional triggers or environmental stimuli, rather than careful consideration. Platforms like Steam leverage this behavior by showcasing what’s popular among friends or within the broader community, reinforcing the impulse to buy during discount periods.

In conclusion, understanding the nuances of price sensitivity, the psychology of discounts, and social influence in digital platforms like Steam provides valuable insights into how consumers engage with pricing and sales strategies. These elements are crucial for developers and marketers aiming to maximize sales during discount periods while maintaining long-term customer satisfaction and retention.

---

**Sources**:
1. "How Price Sensitivity Affects Consumer Buying Behavior," https://stores-goods.com/blog/consumer-behavior/how-price-sensitivity-affects-consumer-buying/.
2. "How Price Sensitivity Analysis Can Reveal Consumer Behavior," https://borderlessaccess.com/blog/understanding-how-customer-behavior-is-shaped-by-price-sensitivity/.
3. "The Psychology of Discounts: 8 Researched-Backed Strategies," https://www.namogoo.com/blog/consumer-behavior-psychology/psychology-of-discounts/.
4. Krishna, Aradhna et al. "A meta-analysis of the impact of price presentation on perceived savings." Journal of Retailing, 78(2), 2002, 101-118. https://doi.org/10.1016/S0022-4359(02)00072-6.
5. Zhang, Mingming, and Shi, Guicheng. "Consumers’ Impulsive Buying Behavior in Online Shopping Based on the Influence of Social Presence." Computational Intelligence and Neuroscience, 2022. https://doi.org/10.1155/2022/6794729.


## Dataset Generality
The distribution of my dataset is representative of the real-world gaming industry, particularly on platforms like Steam, for multiple reasons. First, the dataset includes a diverse range of games, from indie titles to AAA blockbusters, spanning various genres and categories. This reflects the broad spectrum of games available to consumers in the real world, where both small and large developers coexist. The pricing structure, including free-to-play games, heavily discounted titles, and premium-priced games, mirrors the common pricing strategies used in digital gaming marketplaces.

Additionally, the user review system, where games are categorized based on feedback such as “Very Positive” or “Mostly Negative,” is a direct representation of how real consumers engage with games on platforms like Steam. The distribution of review percentages, ranging from overwhelmingly positive to negative, captures the varied experiences players have with different games, aligning with the real-world setting where games appeal to different audiences.

Moreover, the presence of a few highly rated, popular games alongside a larger number of less-known or niche titles reflects the natural long-tail distribution observed in many markets, including gaming. This dataset, with its focus on reviews and pricing, captures the key elements that shape consumer behavior in digital game purchasing, making it an accurate representation of real-world market trends.

## Data Transformations

### Transformation 1
**Description:**  
Renaming columns such as `'title'` to `'game_title'`, `'age_rating'` to `'age_restricted'`, and others for clarity.

**Soundness Justification:**  
This transformation does not affect the underlying data but improves readability. The renaming operation preserves the original intent of each column, ensuring the semantics remain unchanged. It makes the dataset easier to interpret without discarding or altering any usable data.

---

### Transformation 2
**Description:**  
Converting the `'release_date'` column to `datetime` format using `pd.to_datetime()` with `errors='coerce'` to handle invalid formats.

**Soundness Justification:**  
This transformation ensures that the date is properly formatted for time-based analysis. The use of `errors='coerce'` converts invalid dates to `NaT`, preventing the introduction of errors. This does not change the semantics of the data, as invalid dates are safely handled without impacting other fields.

---

### Transformation 3
**Description:**  
Converting `'age_rating'` to boolean (True/False) and renaming it to `'age_restricted'`.

**Soundness Justification:**  
This transformation does not discard data or introduce errors. The boolean conversion makes the data clearer without changing its meaning, as `True`/`False` is a more intuitive representation of whether a game is age-restricted than `0`/`1`.

---

### Transformation 4
**Description:**  
Converting `'original_price'` and `'discounted_price'` to `float` dtype, removing non-numeric characters (₹, commas), and imputing missing values. Free games are set to `0.00`, and missing original prices are set equal to the discounted price.

**Soundness Justification:**  
This transformation ensures consistency in price values without introducing outliers or errors. Free games are accurately handled, and imputing missing original prices with the discounted price preserves the intended meaning of the data. The rupee symbol and commas are safely removed without affecting numerical operations.

---

### Transformation 5
**Description:**  
Converting `'discount_percentage'` to the remaining price percentage (after the discount) by subtracting from 100, and filling missing values with 100 (indicating no discount).

**Soundness Justification:**  
This transformation simplifies the interpretation of discount percentages. Using `100%` for missing values ensures no unintended outliers or data loss occurs. The operation does not change the meaning of the data, as it reflects the full price when no discount is applied.

---

### Transformation 6
**Description:**  
Dropping rows with missing values for `'developer'`, `'publisher'`, `'genres'`, `'categories'`, and `'release_date'`, as these represent less than 1% of the dataset.

**Soundness Justification:**  
Since the missing values are minimal, dropping these rows has an insignificant effect on the dataset. This operation ensures that incomplete records do not introduce bias into the analysis without discarding usable data.

---

### Transformation 7
**Description:**  
Imputing missing values for reviews:
- `'overall_review'` is filled with `'No Reviews'`.
- `'overall_review_count'` is filled with `0`.
- `'overall_positive_review_percentage'` is filled with `-1` to indicate no reviews.

**Soundness Justification:**  
This transformation handles missing review data carefully. Imputing missing reviews with `-1` instead of `0%` prevents misleading interpretation (as `0%` implies all negative reviews, not the absence of reviews). This preserves the data’s meaning while ensuring completeness.

---

### Transformation 8
**Description:**  
Dropping columns like `'content_descriptor_tags'` and `'recent_review'` (with more than 85% missing data), and rows with missing descriptions (`'store_game_description'`).

**Soundness Justification:**  
Since the missing data in these columns is overwhelming, dropping them is necessary to prevent skewed results. This operation retains the dataset’s integrity by keeping only useful, non-empty fields and does not affect the remaining usable data.

---


## Visualizations


---

### Visual 1: Original Price (INR) vs Awards Count
**Analysis:** This scatter plot shows the relationship between the original price of games (in INR) and their award counts. The distribution suggests that most games, regardless of price, tend to have fewer than 10 awards. A few high-priced games achieve significantly higher awards, but the correlation is not strong, indicating that the number of awards is not directly tied to the game's price.

---

### Visual 2: Original Price (INR) vs Positive Review Percentage
**Analysis:** This plot highlights the relationship between a game's price and the percentage of positive reviews. The distribution indicates that games across all price ranges have high positive review percentages, particularly above 60%. This suggests that both low- and high-priced games can have positive reception, with price being a less significant factor in determining the quality of reviews.

---

### Visual 3: Top 20 Genres
**Analysis:** The bar chart visualizes the top 20 most common genres in the dataset. The "Action, Indie" and "Action, Adventure, Indie" genres dominate, highlighting the popularity of action-based indie games. The plot shows the diversity of game genres available on Steam, with a strong leaning toward combinations of indie, action, and adventure genres, which are popular among both players and developers.

---

### Visual 4: Distribution of Overall Review
**Analysis:** This bar chart displays the distribution of various overall review categories, ranging from "Very Positive" to "Overwhelmingly Negative." The majority of games fall into the "Very Positive" category, with a significant number of games also receiving "Mixed" and "Positive" reviews. The lower categories, such as "Very Negative" and "Overwhelmingly Negative," represent a much smaller portion of the dataset. This suggests that most games on Steam are reviewed favorably by the user base, with negative reviews being less common overall.

---

### Visual 5: Distribution of Overall Positive Review Percentage
**Analysis:** This histogram shows the distribution of overall positive review percentages for games. Most games cluster around the higher review ranges (60-100%), indicating that a majority of games have been favorably received. The right skew of the distribution suggests that players tend to rate games positively, with very few games receiving overwhelmingly negative feedback.

---
