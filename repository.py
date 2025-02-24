import uuid
from sqlalchemy.orm import Session
import models, schemas
import utils

# ------------------------
# Users
# ------------------------
def create_user(db: Session, user: schemas.UserCreate):
    """
    Create a new user in the database.

    Args:
        db (Session): The database session.
        user (schemas.UserCreate): The user creation data.

    Returns:
        models.User: The created user.
    """
    hashed_password = utils.get_password_hash(user.password)
    id = utils.generate_uuid()

    db_user = models.User(
        id=id,
        username=user.username,
        hashed_password=hashed_password,
        email=user.email,
        is_supplier=user.is_supplier,
        is_verified=user.is_verified
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int):
    """
    Retrieve a user by their ID.

    Args:
        db (Session): The database session.
        user_id (int): The user ID.

    Returns:
        models.User: The retrieved user.
    """
    return db.query(models.User).filter(models.User.id == user_id).first()

def login_user(db: Session, username: str, password: str):
    """
    Authenticate a user using username and password.

    Args:
        db (Session): The database session.
        username (str): The username address of the user.
        password (str): The plain text password provided by the user.

    Returns:
        models.User: The authenticated user, or None if authentication fails.
    """
    # Query for the user based on username
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return None
    
    # Verify the password using a utility function; 
    # ensure that user.hashed_password exists and contains the hashed password.
    if not utils.verify_password(password, user.hashed_password):
        return None

    return user

# ------------------------
# Posts
# ------------------------
def create_post(db: Session, post: schemas.PostCreate):
    """
    Create a new post in the database.

    Args:
        db (Session): The database session.
        post (schemas.PostCreate): The post creation data.

    Returns:
        models.Post: The created post.
    """
    post_id = utils.generate_uuid()
    db_post = models.Post(
        id=post_id,
        title=post.title,
        description=post.description,
        category=post.category,
        status=post.status,
        requester_id=post.requester_id
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

def get_post(db: Session, post_id: int):
    """
    Retrieve a post by its ID.

    Args:
        db (Session): The database session.
        post_id (int): The post ID.

    Returns:
        models.Post: The retrieved post.
    """
    return db.query(models.Post).filter(models.Post.id == post_id).first()

# ------------------------
# Bids
# ------------------------
def create_bid(db: Session, bid: schemas.BidCreate):
    """
    Create a new bid in the database.

    Args:
        db (Session): The database session.
        bid (schemas.BidCreate): The bid creation data.

    Returns:
        models.Bid: The created bid.
    """
    bid_id = utils.generate_uuid()
    
    db_bid = models.Bid(
        id=bid_id,
        post_id=bid.post_id,
        supplier_id=bid.supplier_id,
        price=bid.price,
        message=bid.message,
        status=bid.status
    )
    db.add(db_bid)
    db.commit()
    db.refresh(db_bid)
    return db_bid

def get_bid(db: Session, bid_id: int):
    """
    Retrieve a bid by its ID.

    Args:
        db (Session): The database session.
        bid_id (int): The bid ID.

    Returns:
        models.Bid: The retrieved bid.
    """
    return db.query(models.Bid).filter(models.Bid.id == bid_id).first()

# ------------------------
# Messages
# ------------------------
def create_message(db: Session, message: schemas.MessageCreate):
    """
    Create a new message in the database.

    Args:
        db (Session): The database session.
        message (schemas.MessageCreate): The message creation data.

    Returns:
        models.Message: The created message.
    """
    message_id = utils.generate_uuid()
    db_message = models.Message(
        id=message_id,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        content=message.content
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_message(db: Session, message_id: int):
    """
    Retrieve a message by its ID.

    Args:
        db (Session): The database session.
        message_id (int): The message ID.

    Returns:
        models.Message: The retrieved message.
    """
    return db.query(models.Message).filter(models.Message.id == message_id).first()

# ------------------------
# Reviews
# ------------------------
def create_review(db: Session, review: schemas.ReviewCreate):
    """
    Create a new review in the database.

    Args:
        db (Session): The database session.
        review (schemas.ReviewCreate): The review creation data.

    Returns:
        models.Review: The created review.
    """
    review_id = utils.generate_uuid()
    db_review = models.Review(
        id=review_id,
        post_id=review.post_id,
        supplier_id=review.supplier_id,
        customer_id=review.customer_id,
        rating=review.rating,
        review=review.review
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

def get_review(db: Session, review_id: int):
    """
    Retrieve a review by its ID.

    Args:
        db (Session): The database session.
        review_id (int): The review ID.

    Returns:
        models.Review: The retrieved review.
    """
    return db.query(models.Review).filter(models.Review.id == review_id).first()

# ------------------------
# Services
# ------------------------
def create_service(db: Session, service: schemas.ServiceCreate):
    """
    Create a new service in the database.

    Args:
        db (Session): The database session.
        service (schemas.ServiceCreate): The service creation data.

    Returns:
        models.Service: The created service.
    """
    service_id = utils.generate_uuid()

    db_service = models.Service(
        id=service_id,
        name=service.name,
        description=service.description
    )
    db.add(db_service)
    db.commit()
    db.refresh(db_service)
    return db_service

def get_service(db: Session, service_id: int):
    """
    Retrieve a service by its ID.

    Args:
        db (Session): The database session.
        service_id (int): The service ID.

    Returns:
        models.Service: The retrieved service.
    """
    return db.query(models.Service).filter(models.Service.id == service_id).first()

# ------------------------
# Supplier Services (Association)
# ------------------------
def add_supplier_service(db: Session, supplier_id: int, service_id: int):
    """
    Associate a supplier with a service.

    Args:
        db (Session): The database session.
        supplier_id (int): The supplier's user ID.
        service_id (int): The service ID.

    Returns:
        models.SupplierService: The created supplier service association.
    """
    supplier_service_id = utils.generate_uuid()
    db_supplier_service = models.SupplierService(
        id=supplier_service_id,
        supplier_id=supplier_id,
        service_id=service_id
    )
    db.add(db_supplier_service)
    db.commit()
    db.refresh(db_supplier_service)
    return db_supplier_service

# ------------------------
# Transactions
# ------------------------
def create_transaction(db: Session, transaction: schemas.TransactionCreate):
    """
    Create a new transaction in the database.

    Args:
        db (Session): The database session.
        transaction (schemas.TransactionCreate): The transaction creation data.

    Returns:
        models.Transaction: The created transaction.
    """
    tramnsaction_id = utils.generate_uuid()
    db_transaction = models.Transaction(
        id=tramnsaction_id,
        post_id=transaction.post_id,
        supplier_id=transaction.supplier_id,
        customer_id=transaction.customer_id,
        amount=transaction.amount
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def get_transaction(db: Session, transaction_id: int):
    """
    Retrieve a transaction by its ID.

    Args:
        db (Session): The database session.
        transaction_id (int): The transaction ID.

    Returns:
        models.Transaction: The retrieved transaction.
    """
    return db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()

# ------------------------
# Appointments
# ------------------------
def create_appointment(db: Session, appointment: schemas.AppointmentCreate):
    """
    Create a new appointment in the database.

    Args:
        db (Session): The database session.
        appointment (schemas.AppointmentCreate): The appointment creation data.

    Returns:
        models.Appointment: The created appointment.
    """
    appointment_id = utils.generate_uuid()
    db_appointment = models.Appointment(
        id=appointment_id,
        post_id=appointment.post_id,
        supplier_id=appointment.supplier_id,
        customer_id=appointment.customer_id,
        appointment_time=appointment.appointment_time
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    return db_appointment

def get_appointment(db: Session, appointment_id: int):
    """
    Retrieve an appointment by its ID.

    Args:
        db (Session): The database session.
        appointment_id (int): The appointment ID.

    Returns:
        models.Appointment: The retrieved appointment.
    """
    return db.query(models.Appointment).filter(models.Appointment.id == appointment_id).first()
