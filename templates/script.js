async function checkSpam() {
  const comment = document.getElementById("comment").value;
  const resultEl = document.getElementById("result");

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comment })
  });

  const data = await response.json();
  resultEl.innerText = `${data.prediction} (Accuracy: ${data.accuracy}%)`;
}
