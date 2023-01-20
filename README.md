<h1> Chess Computer Vision Website </h1>

Here I have created a chess website using Flask. The main point of the website is to upload a picture of a physical chess-board and the CNN model will predict the layout
the board automatically. This is accomplished by using SQLite to store the image as a glob in a database and then transfer that information to the model as an array. The model then transforms the image using a bunch of fancy OpenCV tools (houghlinesP, uncanny-edge detection, contouring, etc). It also uses perspective transformation to realign the board into a more neutral top-down position.

<img src="/current_board.svg" width="500" height="500">

The website also has a user authentication database made using Flask and werkzeug.security with a CSS front-end built using Bootstrap:

<img src="/login.jpg" width="500" height="250">

The main libraries used for this project were:

1. Flask
2. flask_sqlalchemy
3. werkzeug
4. wtforms
5. OpenCV

### Paul Fisher 2022
