# User Analytics in the Telecommunication Industry

## Project Overview
This project aims to provide a detailed analysis of user behavior and performance in the telecommunication industry. It leverages a comprehensive dataset on customer activities, network parameters, and device characteristics to deliver insights that can drive profitability and inform strategic decision-making.

## Key Features
1. **Reusable Code for Data Preparation and Cleaning**: The project includes modular and well-documented code for handling data quality issues, missing values, and outliers, ensuring the data is in a format suitable for analysis.
2. **Estimated Code Complexity**: The key components of the code, including data structures and algorithms, have been profiled to provide estimates of running time and memory requirements.
3. **Scikit-learn Pipeline Integration**: The analysis steps are connected using the scikit-learn pipeline or other forms of chaining, enabling a seamless and efficient workflow.
4. **Streamlit Dashboard**: An interactive Streamlit-based dashboard is developed to present the findings of the user analytics study in a visually appealing and user-friendly manner.
5. **SQL Database as Feature Store**: A SQL database (e.g., PostgreSQL or MySQL) is used as a feature store to store the selected features for dashboard visualization and model training.
6. **Installable via pip**: The project is packaged and installable via pip, allowing for easy deployment and integration into other projects.
7. **Unit Tests and CI/CD**: The project includes comprehensive unit tests with good coverage, and a CI/CD setup using GitHub Actions ensures the codebase's quality and maintainability.
8. **Docker Integration**: A Dockerfile is provided to build the project as a Docker image, facilitating easy deployment and reproducibility.
9. **Adherence to Streamlit Coding Practices**: The Python code follows the style and structure of Streamlit's source code, providing a reference for learning advanced Python programming techniques.

# User Overview Analysis
   - Identify the top 10 handsets used by customers and the top 3 handset manufacturers.
   - Analyze the top 5 handsets per top 3 handset manufacturer and provide interpretations and recommendations for the marketing teams.
   - Aggregate user information, such as the number of sessions, session duration, download, and upload data, for various applications (e.g., Social Media, Google, Email, YouTube, Netflix, Gaming, Other).
   - Perform exploratory data analysis on the user data, including non-graphical and graphical univariate analysis, bivariate analysis, variable transformations, correlation analysis, and dimensionality reduction using principal component analysis.

# User Engagement Analysis
   - Aggregate engagement metrics (session frequency, session duration, and total traffic) per customer and report the top 10 customers for each metric.
   - Normalize the engagement metrics and use k-means clustering (k=3) to classify customers into three engagement groups.
   - Compute and interpret the minimum, maximum, average, and total non-normalized metrics for each engagement cluster.
   - Aggregate user total traffic per application and derive the top 10 most engaged users per application.
   - Plot the top 3 most used applications using appropriate charts.
   - Determine the optimized number of clusters (k) using the elbow method and interpret the user engagement clusters.

# User Experience Analysis
   - Aggregate network performance metrics (average TCP retransmission, average RTT, and average throughput) and handset type per customer.
   - Compute and report the top 10, bottom 10, and most frequent values for TCP retransmission, RTT, and throughput.
   - Analyze the distribution of average throughput and average TCP retransmission per handset type.
   - Perform k-means clustering (k=3) on the user experience metrics and provide a brief description of each cluster.

# User Satisfaction Analysis
   - Assign engagement and experience scores to each user based on the previous analyses.
   - Calculate the satisfaction score as the average of the engagement and experience scores, and report the top 10 most satisfied customers.
   - Build a regression model to predict the satisfaction score of a customer.
   - Perform k-means clustering (k=2) on the engagement and experience scores, and aggregate the average satisfaction and experience scores per cluster.
   - Export the final table containing all user IDs, engagement, experience, and satisfaction scores to a local MySQL database.
   - Provide a model deployment tracking report, including code version, start and end time, source, parameters, metrics, and any output files.

## License
This project is licensed under the [MIT License](LICENSE).

