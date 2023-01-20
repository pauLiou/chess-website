from website import create_app

app = create_app()

if __name__ == '__main__': # only run if its directly from the main.py script
    app.run(debug=True) # debug = automatically rerun the code to avoid manually re-running (turn off in production)

