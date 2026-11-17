
import pymysql
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from dotenv import load_dotenv
from pathlib import Path
import os


# Load project environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def migrate_users():
    mysql_connection = None
    mongo_client = None

    try:
        # 1. Connect to MySQL
        print("Connecting to MySQL...")

        mysql_connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            cursorclass=pymysql.cursors.DictCursor
        )

        print("MySQL connected successfully.")

        # 2. Connect to MongoDB Atlas
        print("Connecting to MongoDB Atlas...")

        mongo_client = MongoClient(
            os.getenv("MONGODB_URI"),
            serverSelectionTimeoutMS=15000
        )

        mongo_client.admin.command("ping")

        database = mongo_client[os.getenv("MONGODB_DATABASE")]
        users_collection = database["users"]

        print("MongoDB connected successfully.")

        # 3. Create unique indexes
        users_collection.create_index(
            "legacy_id",
            unique=True
        )

        users_collection.create_index(
            "normalized_email",
            unique=True
        )

        # 4. Read MySQL accounts
        with mysql_connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    id,
                    full_name,
                    email,
                    password_hash,
                    failed_login_attempts,
                    is_locked,
                    created_at
                FROM users
            """)

            mysql_users = cursor.fetchall()

        print(f"\nFound {len(mysql_users)} MySQL accounts.")

        # 5. Copy accounts into MongoDB
        inserted = 0
        already_present = 0

        for user in mysql_users:
            legacy_id = int(user["id"])
            original_email = user["email"]
            normalized_email = original_email.strip().lower()

            existing = users_collection.find_one({
                "legacy_id": legacy_id
            })

            if existing:
                already_present += 1
                continue

            if users_collection.find_one({
                "normalized_email": normalized_email
            }):
                raise ValueError(
                    f"Email conflict for MySQL account ID {legacy_id}"
                )

            mongo_user = {
                "legacy_id": legacy_id,
                "full_name": user["full_name"],
                "email": original_email,
                "normalized_email": normalized_email,
                "password_hash": user["password_hash"],
                "failed_login_attempts": int(
                    user["failed_login_attempts"]
                ),
                "is_locked": bool(user["is_locked"]),
                "created_at": user["created_at"],
            }

            users_collection.insert_one(mongo_user)
            inserted += 1

            print(f"Migrated account ID: {legacy_id}")

        # 6. Verify migration
        mongo_count = users_collection.count_documents({
            "legacy_id": {"$exists": True}
        })

        print("\n========== MIGRATION REPORT ==========")
        print("MySQL accounts:", len(mysql_users))
        print("New accounts copied:", inserted)
        print("Already migrated:", already_present)
        print("Total migrated accounts:", mongo_count)
        print("======================================")

        if mongo_count != len(mysql_users):
            print("WARNING: Account counts do not match.")
        else:
            print("SUCCESS: All accounts are present in MongoDB.")

    except Exception as error:
        print("\nMIGRATION FAILED:", str(error))
        raise

    finally:
        if mysql_connection:
            mysql_connection.close()

        if mongo_client:
            mongo_client.close()


if __name__ == "__main__":
    migrate_users()
