document
  .getElementById("commentForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const comment = document.getElementById("comment").value;

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ comment: comment }),
    });

    if (response.ok) {
      const data = await response.json();
      document.getElementById("primaryCategory").innerText =
        "Primary Category: " + data.primary_category;
      document.getElementById("secondaryCategory").innerText =
        "Secondary Category: " + data.secondary_category;
      document.getElementById("result").style.display = "block";
    } else {
      alert("Error: Could not classify the comment.");
    }
  });
