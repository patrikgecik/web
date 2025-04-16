<?php

$result = '';

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $text = escapeshellarg($_POST['text']);
    $lang = $_POST['lang'];
    $command = "\"C:\\Users\\Dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe\" predict_bert.py $text $lang";
    $result = shell_exec($command . " 2>&1");
}

?>

<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <title>Toxicita v texte</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #e8f0ff, #f5faff);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px 20px;
            min-height: 100vh;
            margin: 0;
        }

        .container {
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 550px;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 25px;
        }

        .tab-buttons {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
            gap: 10px;
        }

        .tab-buttons button {
            flex: 1;
            padding: 10px;
            font-size: 15px;
            cursor: pointer;
            background: #e0e0e0;
            border: none;
            border-radius: 5px;
            transition: 0.3s;
        }

        .tab-buttons button.active {
            background: #007BFF;
            color: white;
        }

        form {
            display: flex;
            flex-direction: column;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 6px;
            resize: vertical;
            margin-bottom: 15px;
            box-sizing: border-box;
        }

        input[type="submit"] {
            padding: 12px;
            background: #007BFF;
            border: none;
            color: white;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.3s;
        }

        input[type="submit"]:hover {
            background: #005ec2;
        }

        .result {
            margin-top: 25px;
            padding: 15px;
            background: #f1f5ff;
            border-left: 4px solid #007BFF;
            border-radius: 6px;
            font-size: 15px;
            white-space: pre-wrap;
            color: #222;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Detekcia toxicity</h1>
    <div class="tab-buttons">
        <button type="button" class="active" onclick="showForm('sk')">Slovensky</button>
        <button type="button" onclick="showForm('en')">English</button>
    </div>

    <form method="post">
        <input type="hidden" name="lang" id="langInput" value="sk">
        <textarea name="text" id="textInput" placeholder="Zadaj vetu" required></textarea>
        <input type="submit" value="Skontroluj">
    </form>

    <?php if (!empty($result)): ?>
        <div class="result">
            <strong>Výsledok:</strong><br>
            <?php echo nl2br(htmlspecialchars($result)); ?>
        </div>
    <?php endif; ?>
</div>

<script>
    function showForm(lang) {
        document.getElementById('langInput').value = lang;
        document.getElementById('textInput').placeholder = lang === 'sk' ? 'Zadaj vetu' : 'Enter a sentence';
        const buttons = document.querySelectorAll('.tab-buttons button');
        buttons.forEach(btn => btn.classList.remove('active'));
        document.querySelector(`.tab-buttons button[onclick="showForm('${lang}')"]`).classList.add('active');
    }
</script>
</body>
</html>
