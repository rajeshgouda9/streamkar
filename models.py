from sqlalchemy import Column, Integer, String, Text, DateTime, func, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
# Replace with your actual PostgreSQL connection details
DATABASE_URL = "postgresql://postgres:1234@localhost:5432/streamkar"



# Define schema
SCHEMA = "streamkar"

Base = declarative_base()

class Chat(Base):
    __tablename__ = "chat"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)#this should be a foreign key
    input = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_date = Column(DateTime, default=func.now())
    #we can have a new column called bot_type like llm, rag, custom agent
    #if rag then we can have the source as well
    #we should also have request_id to check the logs based on unique request id for each message


class FAQ(Base):
    __tablename__ = "faq"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_date = Column(DateTime, default=func.now())

engine = create_engine(DATABASE_URL)
with engine.connect() as connection:
    connection.execute(text("CREATE SCHEMA IF NOT EXISTS streamkar"))

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
