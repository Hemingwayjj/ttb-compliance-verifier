from rapidfuzz import fuzz

def compare(extracted, approved):

    results = {}

    results["abv"] = (
        extracted["abv"] == approved["abv"]
    )

    results["net_contents"] = (
        extracted["net_contents"]
        == approved["net_contents"]
    )

    score = fuzz.ratio(
        extracted["brand_name"],
        approved["brand_name"]
    )

    results["brand_score"] = score

    return results
