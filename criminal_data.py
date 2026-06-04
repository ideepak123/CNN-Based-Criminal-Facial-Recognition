criminal_database = {

    "deepak": {
        "criminal_id": "CR001",
        "name": "Deepak",
        "crime": "Cyber Fraud",
        "status": "Wanted",
        "last_seen": "Andhra Pradesh"
    },

    "narendra": {
        "criminal_id": "CR002",
        "name": "Narendra",
        "crime": "Identity Theft",
        "status": "Under Investigation",
        "last_seen": "Hyderabad"
    }

}


def get_criminal_details(person_name):

    return criminal_database.get(
        person_name.lower(),
        None
    )