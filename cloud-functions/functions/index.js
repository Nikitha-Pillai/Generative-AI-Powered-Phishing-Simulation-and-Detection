const functions = require("firebase-functions");
const fetch = require("node-fetch");

// HuggingFace Spaces URL
const HF_SPACE_URL = "https://littleprophisher-phishing-backend.hf.space";

// ==========================================================
// CLOUD FUNCTION — triggers when user_feedback is written
// Fires automatically when generation_status == "pending"
// ==========================================================

exports.triggerRetraining = functions.firestore
    .document("user_feedback/{docId}")
    .onWrite(async (change, context) => {

        // Get the new data after write
        const data = change.after.data();

        // Only trigger if generation_status is pending
        if (!data || data.generation_status !== "pending") {
            console.log("Not pending — skipping trigger");
            return null;
        }

        console.log("Pending feedback detected — triggering retraining...");

        try {
            const response = await fetch(
                `${HF_SPACE_URL}/trigger-retraining`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );

            const result = await response.json();
            console.log("Retraining triggered:", result.message);

        } catch (error) {
            console.error("Failed to trigger retraining:", error.message);
        }

        return null;
    });