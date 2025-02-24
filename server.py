from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import repository, schemas
from database import SessionLocal, Base, engine

# Create the tables in the database
Base.metadata.create_all(bind=engine)

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------
# Users Endpoints
# ------------------------
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user.
    """
    return repository.create_user(db=db, user=user)

@app.post("/login", response_model=schemas.User)
def login(login_request: schemas.UserLogin, db: Session = Depends(get_db)):
    """
    Login endpoint for user authentication.
    """
    user = repository.login_user(db=db, username=login_request.username, password=login_request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user

@app.get("/users/{user_id}", response_model=schemas.User)
def get_user(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a user by ID.
    """
    db_user = repository.get_user(db=db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# ------------------------
# Posts Endpoints
# ------------------------
@app.post("/posts/", response_model=schemas.Post)
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db)):
    """
    Create a new post.
    """
    return repository.create_post(db=db, post=post)

@app.get("/posts/{post_id}", response_model=schemas.Post)
def get_post(post_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a post by ID.
    """
    db_post = repository.get_post(db=db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post

# ------------------------
# Bids Endpoints
# ------------------------
@app.post("/bids/", response_model=schemas.Bid)
def create_bid(bid: schemas.BidCreate, db: Session = Depends(get_db)):
    """
    Create a new bid.
    """
    return repository.create_bid(db=db, bid=bid)

@app.get("/bids/{bid_id}", response_model=schemas.Bid)
def get_bid(bid_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a bid by ID.
    """
    db_bid = repository.get_bid(db=db, bid_id=bid_id)
    if db_bid is None:
        raise HTTPException(status_code=404, detail="Bid not found")
    return db_bid

# ------------------------
# Messages Endpoints
# ------------------------
@app.post("/messages/", response_model=schemas.Message)
def create_message(message: schemas.MessageCreate, db: Session = Depends(get_db)):
    """
    Create a new message.
    """
    return repository.create_message(db=db, message=message)

@app.get("/messages/{message_id}", response_model=schemas.Message)
def get_message(message_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a message by ID.
    """
    db_message = repository.get_message(db=db, message_id=message_id)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message

# ------------------------
# Reviews Endpoints
# ------------------------
@app.post("/reviews/", response_model=schemas.Review)
def create_review(review: schemas.ReviewCreate, db: Session = Depends(get_db)):
    """
    Create a new review.
    """
    return repository.create_review(db=db, review=review)

@app.get("/reviews/{review_id}", response_model=schemas.Review)
def get_review(review_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a review by ID.
    """
    db_review = repository.get_review(db=db, review_id=review_id)
    if db_review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return db_review

# ------------------------
# Services Endpoints
# ------------------------
@app.post("/services/", response_model=schemas.Service)
def create_service(service: schemas.ServiceCreate, db: Session = Depends(get_db)):
    """
    Create a new service.
    """
    return repository.create_service(db=db, service=service)

@app.get("/services/{service_id}", response_model=schemas.Service)
def get_service(service_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a service by ID.
    """
    db_service = repository.get_service(db=db, service_id=service_id)
    if db_service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return db_service

# ------------------------
# Supplier Services (Association) Endpoint
# ------------------------
@app.post("/supplier_services/")
def add_supplier_service(supplier_id: str, service_id: str, db: Session = Depends(get_db)):
    """
    Associate a supplier with a service.
    """
    return repository.add_supplier_service(db=db, supplier_id=supplier_id, service_id=service_id)

# ------------------------
# Transactions Endpoints
# ------------------------
@app.post("/transactions/", response_model=schemas.Transaction)
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    """
    Create a new transaction.
    """
    return repository.create_transaction(db=db, transaction=transaction)

@app.get("/transactions/{transaction_id}", response_model=schemas.Transaction)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a transaction by ID.
    """
    db_transaction = repository.get_transaction(db=db, transaction_id=transaction_id)
    if db_transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return db_transaction

# ------------------------
# Appointments Endpoints
# ------------------------
@app.post("/appointments/", response_model=schemas.Appointment)
def create_appointment(appointment: schemas.AppointmentCreate, db: Session = Depends(get_db)):
    """
    Create a new appointment.
    """
    return repository.create_appointment(db=db, appointment=appointment)

@app.get("/appointments/{appointment_id}", response_model=schemas.Appointment)
def get_appointment(appointment_id: str, db: Session = Depends(get_db)):
    """
    Retrieve an appointment by ID.
    """
    db_appointment = repository.get_appointment(db=db, appointment_id=appointment_id)
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return db_appointment
