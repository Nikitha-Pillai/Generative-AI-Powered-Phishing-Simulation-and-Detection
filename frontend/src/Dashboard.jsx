import React, { useEffect, useState } from "react";
import "./App.css";

// ── API URL — HuggingFace Spaces
const API_URL = "https://littleprophisher-phishing-backend.hf.space";

function Dashboard() {
  const [emails, setEmails] = useState([]);
  const [expandedIndex, setExpandedIndex] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/api/emails`)
      .then((res) => res.json())
      .then((data) => setEmails(data.reverse()))
      .catch((err) => console.error("Error fetching emails:", err));
  }, []);

  // Convert LIME object {word: score} to array [{word, score}]
  const parseLime = (lime_explanation) => {
    if (!lime_explanation) return [];
    if (Array.isArray(lime_explanation)) return lime_explanation;
    return Object.entries(lime_explanation).map(([word, score]) => ({
      word,
      score,
    }));
  };
  const cleanEmailContent = (text) => {
  if (!text) return "";

  if (text.trim().startsWith("<")) {
    const doc = new DOMParser().parseFromString(text, "text/html");
    doc.querySelectorAll("style, script, head").forEach(el => el.remove());
    text = doc.body.innerText || doc.body.textContent || "";
  }

  return text
    .replace(/&#847;|&zwnj;|&#8199;|&#65279;|&#173;|&#8203;|&#8204;|&#8205;/g, "")
    .replace(/[\u00AD\u200B\u200C\u200D\uFEFF\u034F\u2007]/g, "")
    .replace(/\s*<\s*$/gm, "")
    .replace(/[-]{5,}/g, "")
    .replace(/Forwarded message/g, "")
    .replace(/From:.*\n?/g, "")
    .replace(/Date:.*\n?/g, "")
    .replace(/To:.*\n?/g, "")
    .replace(/Subject:.*\n?/g, "")
    .replace(/^\s*\*\s*$/gm, "")
    .replace(/^\s*[•\-\*]\s*/gm, "")
    .replace(/^\s+$/gm, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
};
  return (
    <div className="app">
      <h1 className="title">🛡️ AI Phishing Detection Dashboard</h1>

      <div className="table-container">
        {emails.length === 0 ? (
          <p className="empty">No emails found.</p>
        ) : (
          <table className="email-table">
            <thead>
              <tr>
                <th>Subject</th>
                <th>Prediction</th>
                <th>Confidence</th>
              </tr>
            </thead>

            <tbody>
              {emails.map((email, index) => (
                <React.Fragment key={email.email_id}>
                  <tr
                    className={`row ${
                      expandedIndex === index ? "active" : ""
                    }`}
                    onClick={() =>
                      setExpandedIndex(expandedIndex === index ? null : index)
                    }
                  >
                    <td className="subject">
                      {email.subject || "No Subject"}
                    </td>

                    <td className="center">
                      <span
                        className={
                          email.prediction === "Phishing"
                            ? "badge phishing"
                            : "badge legit"
                        }
                      >
                        {email.prediction}
                      </span>
                    </td>

                    <td className="center confidence">
                      {email.confidence
                        ? (email.confidence * 100).toFixed(2)
                        : "0.00"}
                      %
                    </td>
                  </tr>

                  {expandedIndex === index && (
                    <tr className="expanded-row">
                      <td colSpan="3">
                        <strong>Email Body:</strong>
                        <p style={{ whiteSpace: "pre-line" }}>{cleanEmailContent(email.content)}</p>


                        <br />

                        <strong>AI Explanation (LIME):</strong>

                        {parseLime(email.lime_explanation).length > 0 ? (
                          <div className="lime-box">
                            {parseLime(email.lime_explanation)
                              .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
                              .map((item, i) => {
                                const v = parseFloat(item.score);
                                return (
                                  <span
                                    key={i}
                                    className={
                                      v > 0.001
                                        ? "lime-word negative"
                                        : v < -0.001
                                        ? "lime-word positive"
                                        : "lime-word neutral"
                                    }
                                  >
                                    {item.word} ({item.score.toFixed(2)})
                                  </span>
                              );
                              })
                              }
                          </div>
                        ) : (
                          <p className="no-lime">
                            No explanation available.
                          </p>
                        )}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default Dashboard;