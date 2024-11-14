# CS_548_Project
The goal of this project is to take Movie Data from the Letterboxed Kaggle Dataset and use ML algorithims to be able to determine a movies Letterboxed rating based on certain input parameters
https://www.kaggle.com/datasets/gsimonx37/letterboxd/data?select=themes.csv


**Listed Below are the steps weve taken while going throught the project**

Movie DataSet Work

- Cant use small dataset unless want to organize full data set
- Start by creating one master csv
- Try and do it excel with an if statement, basically if id = 1 then add other two columns, continued on to 2. If there is no 2, then add a movie id of 2 and leave next to columns blank. Then continue to 3. Etc.
-Dont think we can use the popularity column since it changes often. Data wont be up to date and doesn't work for what we want

45438 is off by 45448, so ten spots
Up to 42093 all good


-Switching datasets, after attempting to use first dataset, found that too many data points are missing their budget as well as revenue, amount of points goes from 45000 to 5000, and still many of those have 0 listed as a field. Going to move to a dataset that doesn't have budget for the time being

- Actors.csv -- has about 20 actors for each movie, could cause problems
- Countries.csv -- some movies are shot in multiple countries
- crew.csv -- similar stuff, also too much listed crew, can reduce to just directors
- genre.csv -- movies have multiple genres
- language.csv -- some movies have language, others have primary language 
- movies.csv -- one row per movie, title, date, tagline, description, length, rating/5
- posters.csv -- just has links to posters for movie, useless
- releases.csv -- which country the movie released in, probably wont help, probably
- studios.csv -- which studies helped produce the movie, can be 1-10
- theme.csv -- could be a really interesting csv to use, movies themes listed here, seems like there are roughly 20-25 themes, so we could have them choose which 3 themes fit the movie best

- Excluding language due to assumption of English for rating purposes. excluding posters and releases as they won't help as much for this data collection. Also excluding countries for now as this data could be useful, but just for simplicity sake isn't being used

- After going through the themes csv there's a little too many different options for this project. There's 109 in total but could cause some complex problems that are unneeded for this.

- 91000 points that have Letterboxed ratings, makes sense to remove all points that don't have ratings

- 75822 Points left after removing all points that don't have data in either genre, actor, director, or studio

- Removing points that are over 3.5 hours, as most of them are tv shows, also removing points under 1 hour as they are short films and skew our data

- After removing and filtering, we are left with 65247 points

- Then went into splitting values into separate columns. This is in order to keep the code somewhat simple. So now we have actor_1,actor_2,actor_3 etc. This is fine it just affects the model somewhat since now if you put an Actor in for actor_1 its not universally checking all three actor columns. 

- Had to encode Director, Actor, Genre, and Studio to be numerical values. This way the ml model can actually interpret this data instead of strings that it wouldn't be able to understand
