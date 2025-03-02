from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
from typing import List, Generator


import repository, schemas, models
from database import SessionLocal, Base, engine, time
from pipeline.rag_pipeline import RAGPipeline
from pipeline.data_storing.supplier_pdf_ingestion import ingest_supplier_pdf

# Create the tables in the database
Base.metadata.create_all(bind=engine)

# Initialize your pipeline once (global or inside some factory)
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://...")
DB_NAME = "my_rag_db"
COLLECTION_NAME = "chunks"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

rag_pipeline = RAGPipeline(
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    openai_api_key=OPENAI_API_KEY,
    routing_method="llm"  # or "keyword", "semantic", etc.
)
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

# Dependency to get DB session (Dat)
# def get_db() -> Session:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

def get_db() -> Generator[Session, None, None]:
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
# Huy's: update_post   
# ------------------------
# -------------- Supplier PDF Upload --------------
@app.post("/suppliers/{supplier_id}/upload_pdf")
def upload_supplier_pdf(
    supplier_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    1) Validate user is_supplier
    2) Save PDF
    3) Call ingest_supplier_pdf(...) -> auto-detect role
    4) (Optional) store the role in Postgres or link to a Service
    """
    # 1) Validate user
    user = repository.get_user(db, user_id=supplier_id)
    if not user:
        raise HTTPException(status_code=404, detail="Supplier not found")
    if not user.is_supplier:
        raise HTTPException(status_code=400, detail="User is not a supplier")

    # 2) Save PDF locally
    folder = "dummy_resumes"
    os.makedirs(folder, exist_ok=True)
    pdf_path = os.path.join(folder, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    # 3) Ingest with pipeline
    detected_role = ingest_supplier_pdf(rag_pipeline, pdf_path, supplier_id)

    # 4) Optionally store the role in Postgres. E.g.:
    user.detected_role = detected_role
    db.commit()
    # Or if you have a services table, you can link them
    service = repository.get_service_by_name(db, detected_role)
    if not service:
        service = repository.create_service(db, schemas.ServiceCreate(name=detected_role, description=""))
    repository.add_supplier_service(db, supplier_id, service.id)

    return {
        "detail": "PDF uploaded & embedded successfully",
        "supplier_id": supplier_id,
        "detected_role": detected_role
    }
    
@app.post("/search_for_supplier")
def search_for_supplier(
    query: str = Body(...),
    requester_id: str = Body(...), 
    day_of_week: str = Body(None),
    start_time: str = Body(None),
    end_time: str = Body(None),
    db: Session = Depends(get_db)
):
    """
    AI-assisted search. 
    1) Use RAG pipeline to get top matches from vector store (Mongo)
    2) For each chunk doc, map supplier_id -> user record in Postgres
    3) Filter out if user is inactive or unverified (optional)
    4) Return final list of suppliers with relevant info
    """
    # 1) RAG pipeline search
    # 'rag_pipeline' is the instance you created at the top of server.py
    # e.g. rag_pipeline = RAGPipeline(...)
    chunk_matches = rag_pipeline.rag_search(query, top_k=10)

    # chunk_matches is a list of docs like:
    # [
    #   {
    #     "_id": "supplier123_chunk_0",
    #     "supplier_id": "supplier123",
    #     "service_role": "personal trainer",
    #     "chunk_text": "Bob has 5 years experience in fitness, specialized in weight loss...",
    #     "score": 0.92,
    #     ...
    #   },
    #   ...
    # ]
    
    chunk_matches = rag_pipeline.rag_search(query, top_k=10)
    if not chunk_matches:
        # create a new post
        new_post = repository.create_post(db, schemas.PostCreate(
            title="User's request: " + query[:30],
            description=query,
            category="general",
            status="open",
            requester_id=requester_id
        ))
        return {
            "results": [],
            "summary": "No direct matches found. An open request was created.",
            "post_id": new_post.id
        }

     # 2) Build a list of dicts
    supplier_entries = []
    for match in chunk_matches:
        supplier_id = match.get("supplier_id")
        if not supplier_id:
            continue

        # Get user
        user_obj = repository.get_user(db, user_id=supplier_id)
        if not user_obj:
            continue

        # Score from pipeline
        vector_score = float(match.get("score", 0))

        # is_verified
        verified_flag = user_obj.is_verified

        # rating
        avg_rating = repository.get_supplier_avg_rating(db, supplier_id)

        # availability?
        # only check if user provided a day_of_week, start_time, end_time
        if day_of_week and start_time and end_time:
            # parse them if needed
            # we'll assume they come in "HH:MM" format
            st = time.fromisoformat(start_time)  # e.g. "09:00"
            et = time.fromisoformat(end_time)
            is_available = repository.is_supplier_available(db, supplier_id, day_of_week, st, et)
        else:
            is_available = False  # or True if you want default

        # store in a dict
        supplier_entries.append({
            "supplier_id": supplier_id,
            "username": user_obj.username,
            "score": vector_score,
            "final_score": vector_score + (avg_rating / 10.0),  # or any other formula
            "is_verified": verified_flag,
            "avg_rating": avg_rating,
            "is_available": is_available,
            "profile_chunk": match.get("chunk_text", ""),
            "service_role": match.get("service_role", "unknown")
        })

    # 3) Separate verified vs unverified
    verified_list = [s for s in supplier_entries if s["is_verified"]]
    unverified_list = [s for s in supplier_entries if not s["is_verified"]]

    # 4) Sort verified:
    # Priority: 
    # (a) score desc
    # (b) is_available = True first
    # (c) avg_rating desc
    # We'll do a single sort with a custom key:
    # Python sort is ascending, so we invert some values.
    # is_available => True is bigger than False => we can convert it to int (True=1,False=0)
    def verified_sort_key(x):
        return (
            -x["final_score"],            # descending
            int(x["is_available"]), # True=1 => put first
            -x["avg_rating"]        # descending
        )
    verified_sorted = sorted(verified_list, key=verified_sort_key, reverse=False) 
    # (We used negative in the tuple for 'score' and 'avg_rating' so we can keep reverse=False
    # but you can also do reverse=True if you invert the logic carefully.)

    # 5) Sort unverified similarly, just they appear after verified
    def unverified_sort_key(x):
        return (
            -x["final_score"],
            int(x["is_available"]),
            -x["avg_rating"]
        )
    unverified_sorted = sorted(unverified_list, key=unverified_sort_key, reverse=False)

    # 6) Combine
    final_sorted = verified_sorted + unverified_sorted
    
    from pipeline.output.structured_output import ask_chatgpt_structured

    # 6.5) Call ChatGPT for structured output
    summary_response = ask_chatgpt_structured(
        user_query=query,
        retrieved_docs=final_sorted,
        openai_api_key=rag_pipeline.openai_api_key,  # or pipeline uses global key
        method="pydantic"  # or "function_calling"
    )

    # 7) Return
    return {"results": final_sorted, "summary": summary_response}

#  ------------------------
#  Supplier Availability Endpoint
#  ------------------------

@app.post("/suppliers/{supplier_id}/availability", response_model=schemas.SupplierAvailability)
def create_availability_for_supplier(
    supplier_id: str,
    availability: schemas.SupplierAvailabilityCreate,
    db: Session = Depends(get_db)
):
    user = repository.get_user(db, user_id=supplier_id)
    if not user or not user.is_supplier:
        raise HTTPException(status_code=400, detail="User is not a valid supplier")

    # They must match to avoid mismatch errors
    if availability.supplier_id != supplier_id:
        raise HTTPException(status_code=400, detail="supplier_id mismatch in body vs path")

    return repository.create_supplier_availability(db, availability)

@app.get("/suppliers/{supplier_id}/availability", response_model=List[schemas.SupplierAvailability])
def list_availabilities_for_supplier(
    supplier_id: str,
    db: Session = Depends(get_db)
):
    user = repository.get_user(db, user_id=supplier_id)
    if not user or not user.is_supplier:
        raise HTTPException(status_code=400, detail="User is not a valid supplier")

    return repository.get_supplier_availabilities(db, supplier_id)


# ------------------------
# Bids Endpoints
# ------------------------
@app.post("/bids/", response_model=schemas.Bid)
def create_bid(bid: schemas.BidCreate, db: Session = Depends(get_db)):
    """
    Create a new bid. Supplier offers to help on a post.
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
    # Optionally check if the supplier is available in that slot
    # or if the user has existing appointments.
    # Example blueprint:
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

# ------------------------
# Open post endpoint (Suppliers dashboard)
# ------------------------
@app.get("/posts/open")
def get_open_posts(db: Session = Depends(get_db)):
    # Return posts that are open
    open_posts = db.query(models.Post).filter(models.Post.status == "open").all()
    return open_posts

@app.get("/posts/{post_id}/bids")
def list_bids_for_post(post_id: str, db: Session = Depends(get_db)):
    """
    Return all bids for a given post, sorted by rating descending.
    """
    # 1) get all bids from repository
    bids = db.query(models.Bid).filter(models.Bid.post_id == post_id).all()

    # 2) for each bid, gather rating, is_verified, etc.
    results = []
    for b in bids:
        supplier = repository.get_user(db, b.supplier_id)
        if not supplier:
            continue
        avg_rating = repository.get_supplier_avg_rating(db, supplier.id)

        results.append({
            "bid_id": b.id,
            "supplier_id": b.supplier_id,
            "supplier_name": supplier.username,
            "supplier_rating": avg_rating,
            "supplier_verified": supplier.is_verified,
            "price": str(b.price),
            "message": b.message
        })

    # 3) sort by rating desc (maybe also verified first)
    results_sorted = sorted(
        results,
        key=lambda x: (not x["supplier_verified"], -x["supplier_rating"])
    )
    # Explanation: 
    # 'not x["supplier_verified"]' => verified = false => True => goes last
    # '-x["supplier_rating"]' => higher rating first

    return results_sorted

@app.post("/bids/{bid_id}/accept")
def accept_bid(bid_id: str, db: Session = Depends(get_db)):
    """
    Accept a specific bid. This typically changes the post status to 'accepted',
    the bid status to 'accepted', etc. Then the user can proceed to scheduling.
    """
    db_bid = repository.get_bid(db, bid_id=bid_id)
    if not db_bid:
        raise HTTPException(404, "Bid not found")

    # Mark the bid as 'accepted'
    db_bid.status = "accepted"
    db.commit()
    db.refresh(db_bid)

    # Optionally mark the post as 'accepted' or 'in_progress'
    post = repository.get_post(db, db_bid.post_id)
    if post:
        post.status = "accepted"
        db.commit()

    return {"detail": "Bid accepted", "bid_id": bid_id, "post_id": post.id if post else None}
