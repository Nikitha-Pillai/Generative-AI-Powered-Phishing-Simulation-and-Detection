import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Dashboard from "./Dashboard";
import Feedback from "./Feedback";

function App() {
  return (
    <Router>

      <nav style={{marginBottom:"20px"}}>
        <Link to="/">Dashboard</Link> |{" "}
        <Link to="/review">Review Emails</Link>
      </nav>

      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/review" element={<Feedback />} />
      </Routes>

    </Router>
  );
}

export default App;