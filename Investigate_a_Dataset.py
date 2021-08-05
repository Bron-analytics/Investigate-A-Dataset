#!/usr/bin/env python
# coding: utf-8

# # Project: The Movie Database (TMDb) Data Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# 
# ### Data Overview
# 
# > The data is collected from The Movie Databese (TMDb) API. The product uses the TMDb API but is not endorsed or certified by TMDb. The API provides access to data on many movies, actors and actresses, crew members, and TV shows as well as budgets and revenues.
# This dataset contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. We shall analyze this data to dig into some of the important industry questions like what can we say about the success of a movie before it is released?
# 
# > Note that budget_adj (budget adjusted) and revenue_adj (revenue adjusted) columns show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.
# 
# ### Relevant questions we could ask about the data
# 
# > 1. Which genres are most popular from year to year?
# > 2. What kinds of properties are associated with movies that have high revenues?
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style ('darkgrid')

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties
# 
# > **Features And Counts**
# > 1. Samples (rows) = 10866
# > 2. Variables(columns) = 21
# > 3. Duplicate rows = 1 
# > 4. Rows with missing data = 7930

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
df = pd.read_csv('tmdb_movies.csv')
df.head()

#   types and look for instances of missing or possibly errant data.


# In[3]:


df.tail(20)


# In[4]:


# exploring the overall dataset to discover fixes
df.describe()


# In[5]:


# exploring the overall dataset with visuals
df.hist(figsize=(15, 12));


# > release_year and vote_average are good variables to analyze this dataset because of their step-like skewness

# In[6]:


# to find number of samples, number of columns, datatypes of the variables, and missing values 
df.info()


# 
# ### Data Cleaning 
# 

# In[7]:


# to find number of duplicate rows
sum(df.duplicated())


# In[8]:


# to find number of unique values
df.nunique()


# #### 1. Cleaning Column Labels
# > **Dropping extraneous columns** 
# 
# > I dropped features that that are not relevant to the questions for my analysis.
# 
# > **imdb_id:** These values are not necessary in analyzing information within the table.
# 
# > **homepage, tagline, overview, keywords:** These values are unique to each row and cannot be used to make comparisons across multiple rows.
# 
# > **budget, revenue:** Since the **'budget_adj' and 'revenue_adj'** columns adjust the budget and revenue values in terms of 2010 dollars and will therefore allow for more accurate comparisons and analysis, these columns are not needed.
# 
# > **release_date:** I will be using release year to analyze this data, so this more specific column is not needed.

# In[9]:


# using pandas' drop function to drop columns from the dataset: 'id', 'imdb_id', 'homepage', 'tagline', 'keywords', 'overview', 'release_date'
df.drop(['imdb_id', 'homepage', 'tagline', 'keywords', 'overview', 'budget', 'revenue', 'release_date'], axis=1, inplace=True)


# In[10]:


df.head()


# In[11]:


df.info()


# #### 2. Fixing Data Types
# > I fixed **revenue_adj** and **budget_adj** datatypes 
# > 
# >Convert from float to int

# In[12]:


df['revenue_adj'] = df['revenue_adj'].astype(int)
df['budget_adj'] = df['budget_adj'].astype(int)


# In[13]:


df.head()


# #### 3. Rename Data Columns
# > Renaming **budget_adj and revenue_adj** to new **budget and revenue** to make the dataset well labelled

# In[14]:


#rename columns to make variables have good attributes and proper names
df.rename(columns = {'budget_adj':'budget', 'revenue_adj':'revenue'}, inplace=True)
df.head()


# In[15]:


df.info()


# #### 4. Drop Nulls and Dedupe
# > I dropped any rows that contained missing values and dropped any duplicate rows in the dataset
# 

# In[16]:


# view missing value count for each feature in the dataset
df.isnull().sum()


# > There are no missing values for all datatypes that are int and float

# In[17]:


df.dropna(inplace=True)
df.info()


# In[18]:


# print numbers of duplicates in the dataset
print(df.duplicated().sum())


# In[19]:


# drop duplicates 
df.drop_duplicates(inplace=True)


# In[20]:


# print again number of duplicates to confirm dedupe
print(df.duplicated().sum())


# In[21]:


df.info()


# In[22]:


df.shape


# #### 5. Creating Column hybrids
# > I transform the **genres** column to make it easier to work with

# In[23]:


# turn string into list
df['genres'] = df['genres'].str.split('|')
df.head()


# In[24]:


# create a new dataframe with genres listed out in rows
df_gen = df.genres.apply(pd.Series)
df_gen.head()


# #### 6. Merging Datasets
# > I combined the new dataframe with the main dataframe to have a single dataframe for analysis
# 
# > **genres** was originally presented as a string of genre types separated by a **'|'**. This makes it difficult to narrow down movies by one specific genre which is what the first question is all about. First, I turned the strings into a list and then separated the items into their own columns in a new table. Now, I will merge this dataframe with the original dataframe.
# >
# > After that, I will pivot the genre values so that they are presented in their own rows instead of columns. And also, I will remove the original **genres** row and the **variable** row that will be created by the use of the **melt() function** below.
# 

# In[25]:


# merge both dataframes
df_combined = df.merge(df_gen, left_index=True, right_index = True)
df_combined.head()


# In[26]:


# pivot genres from columns into rows
df_combined = df_combined.melt(id_vars=['id' ,'popularity','original_title','cast','director','runtime','genres','production_companies','vote_count','vote_average','release_year','budget','revenue'],value_name="genre")
df_combined.head()


# In[27]:


#remove unnecessary columns
df_combined.drop(['genres','variable'],axis=1,inplace=True)
df_combined.head()


# ### 7. Drop Nulls 
# > Finally, I am going to drop all rows with null values again. I now have two dataframes, **df** and **df_combined**. The first table is the original table of 9,772 rows and the second is the new table where each movie is presented in a row for every genre it is associated with.

# In[28]:


# remove rows with null values
df_combined.dropna(inplace=True)


# In[29]:


df_combined.info()


# In[30]:


df_combined.shape


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.
# 
# > In this step, I will explore the data to answer the below question and plot different visualizations to identify patterns and dependencies.
# 
# ### Research Question 1: Which genres are most popular from year to year? 
# > There are two variables that can be used to understand the popularity of genres of a movie, 'popularity' and 'vote_average'. I compare these columns with each other from year to year to identify which one will be more helpful in answering my question.
# >
# > Using histograms, bar charts, box plots and scatterplots to explore the cleaned dataset

# In[31]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df_combined.plot(x='vote_average', y='popularity',kind='scatter')
plt.title('Vote Average vs Popularity')
plt.xlabel('Vote Average')
plt.ylabel('Popularity')
plt.show()


# In[32]:


df_combined['vote_average'].plot(kind='box')
plt.show()


# In[33]:


df['popularity'].plot(kind='box')
plt.show()


# > I want to first check to make sure that both **popularity** and **vote_average** represent the same thing. The scatter plot above illustrates that both columns are positively correlated. However, as shown by the box plots, **vote_average** is more evenly distributed and lacks any outliers. Nonetheless, all following analysis will be done with **vote_average** as well as **popularity**.

# In[34]:


df_genre = df_combined.groupby('genre').mean()
df_genre = df_genre.sort_values('vote_average')
df_genre
plt.bar(df_genre.index, df_genre['vote_average'], color = 'blue')
plt.xticks(rotation='90')
plt.title('Vote Average by Genre')
plt.ylabel('Vote Average')
plt.xlabel('Genre')
plt.show()


# > To compare vote averge by genre, I used the new dataframe I made while cleaning the data (**df_combined**) to create another dataframe (**df_genre**) that displays the mean value of all columns aggregated by genre type. Then I plotted the **vote average mean** on a bar chart for each genre.
# 
# > The chart shows that the lowest rated movies are **Horror**, **Science Fiction**, and **TV Movie**, while the highest rated are **Documentary**, **History**, and **Music**.

# Now, let's check for the most popular genre and their year

# In[35]:


df_combined['popularity'].describe()


# In[36]:


# use groupby to get popular genres
genre_pop = df_combined.groupby(['release_year', 'genre'], as_index=False)['popularity'].mean()


# In[37]:


# Plot the The Popularity of each genre in the dataset
plt.figure(figsize=[24,12])
sns.barplot(data=genre_pop.sort_values(by='popularity', ascending=False), x='genre', y='popularity')
plt.xlabel('Genre')
plt.ylabel('Popularity')
plt.title('The Popularity of each Genre in TMDB Movies')
plt.show()


# From the combined dataset, **Adventure**, **Animation** and **Fantasy** are the top 3 popular genres

# In[38]:


# use query to determine the most popular genres
most_popular = df_combined.query('popularity > 10')


# In[39]:


# plot the most popular genre (i.e, genres with popularity higher than 10)
most_popular.plot(x='genre', y='popularity', kind='bar', figsize=(8,4))
plt.title('Most Popular Genres in TMDb Movies')
plt.xlabel('Genres')
plt.ylabel('Popularity')
plt.show()


# **Action**, **Adventure**, **Drama**, **Science Fiction** and **Thriller** are the most popular genres year by year

# In[40]:


# print most popular genres and their release year
print(most_popular.groupby(['release_year', 'genre'])['popularity'].mean())


# In[41]:


# check the most popular genre for each year
most_popular.groupby(['release_year', 'genre'])['popularity'].mean().plot(kind='bar')
plt.title('Most Popular Genre by Year')
plt.xlabel('Release Year and Genre')
plt.ylabel('Popularity')
plt.show()


# And again, this shows that **Action**, **Adventure**, **Drama** ,**Science Fiction** and **Thriller** are the most popular genres in their grossing years

# ### Research Question 2: What kinds of properties are associated with movies that have high revenues? 

# In[42]:


# use query to determine high revenue movies
high_revenue = df_combined.query('revenue > revenue.mean()')
low_revenue = df_combined.query('revenue <= revenue.mean()')


# In[43]:


# plot to compare high revenue with low revenue movies
low_revenue.vote_average.plot.hist(color='orange', alpha=0.5, label='Low')
high_revenue.vote_average.plot.hist(color='green', alpha=0.5, label='High')
plt.title('Revenue by Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue Counts')
plt.legend()
plt.show()


# It is clear that we have more low revenue movies from the dataset. High revenue skewed more to the right while low revenue skewed more to the left. This shows that high revenue movies gained higher rating than the low revenue movies.
# Therefore, **vote_average** is one of the properties associated with high revenue movies

# In[44]:


# check the association of high revenue movies with their release year
low_revenue.release_year.plot.hist(color='orange', alpha=0.5, label='Low')
high_revenue.release_year.plot.hist(color='green', alpha=0.5, label='High')
plt.title('Revenue by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Revenue Counts')
plt.legend()
plt.show()


# In[45]:


# check the association of high revenue movies with budget
low_revenue.budget.plot.hist(color='orange', alpha=0.5, label='Low')
high_revenue.budget.plot.hist(color='green', alpha=0.5, label='High')
plt.title('Revenue by Budget')
plt.xlabel('Budget')
plt.ylabel('Revenue Counts')
plt.legend()
plt.show()


# High revenue movies are favoured by high budget as well. 
# Therefore, **budget** is also associated with high revenue movies

# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Tip**: Finally, summarize your findings and the results that have been performed. Make sure that you are clear with regards to the limitations of your exploration. If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# > There are many qualities about movies that make them unique from one another. In this project, I was able to analyze these qualities and identify which properties are associated with movie popularity and high revenue.
# 
# > After cleaning and trimming the dataset by removing unnecessary, null, and duplicated values, I created a secondary table that broke each movie down into the separate genre it falls under.
# Then I plotted a few charts to assess what will be used as the dependent variable, popularity and vote average. 
# 
# ### Findings
# 
# 
# 1. **Action**, **Adventure**, **Drama**, **SCience Fiction** and **Thriller** are the most popular genres year by year
# 2. **Action**, **Adventure**, **Drama**, **SCience Fiction** and **Thriller** are the most popular genres in their grossing year
# 3. **vote_average**, **release_year** and **budget** are the properties associated with high revenue movies
# 
# 
# > Since my analysis only illustrates correlation between variables, it does not definitively conclude whether any trait can predict the popularity and revenue status of a movie. That would require deeper statistical analysis that was not performed in this project.
# 
# ## Submitting your Project 
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[46]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

