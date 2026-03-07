use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const ROMAN_MAP: [(&str, i64); 13] = [
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
];

#[pyfunction(signature = (value=None))]
fn to_roman_scalar(value: Option<i64>) -> PyResult<Option<String>> {
    let Some(mut number) = value else {
        return Ok(None);
    };

    if !(1..4000).contains(&number) {
        return Err(PyValueError::new_err("Number out of range (1-3999)"));
    }

    let mut result = String::new();
    for (roman, numeral) in ROMAN_MAP.iter() {
        while number >= *numeral {
            result.push_str(roman);
            number -= *numeral;
        }
    }

    Ok(Some(result))
}

#[pyfunction(signature = (roman=None))]
fn from_roman_scalar(roman: Option<String>) -> PyResult<Option<i64>> {
    let Some(roman_value) = roman else {
        return Ok(None);
    };

    let mut index = 0usize;
    let chars: Vec<char> = roman_value.chars().collect();
    let mut total = 0i64;

    while index < chars.len() {
        if index + 1 < chars.len() {
            let pair = format!("{}{}", chars[index], chars[index + 1]);
            if let Some((_, value)) = ROMAN_MAP.iter().find(|(key, _)| *key == pair) {
                total += *value;
                index += 2;
                continue;
            }
        }

        let single = chars[index].to_string();
        if let Some((_, value)) = ROMAN_MAP.iter().find(|(key, _)| *key == single) {
            total += *value;
            index += 1;
            continue;
        }

        return Err(PyValueError::new_err("Invalid Roman numeral"));
    }

    Ok(Some(total))
}

fn number_word_value(token: &str) -> Option<i64> {
    match token {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => None,
    }
}

fn parse_word_number(text: &str) -> Result<i64, String> {
    let normalized = text.trim().to_lowercase().replace('-', " ");
    if normalized.is_empty() {
        return Err("Invalid number word".to_string());
    }

    if let Ok(value) = normalized.parse::<i64>() {
        return Ok(value);
    }

    let mut total = 0_i64;
    let mut current = 0_i64;

    for token in normalized.split_whitespace() {
        if token == "and" {
            continue;
        }

        if let Some(value) = number_word_value(token) {
            current += value;
            continue;
        }

        if token == "hundred" {
            if current == 0 {
                current = 1;
            }
            current *= 100;
            continue;
        }

        let scale = match token {
            "thousand" => Some(1_000),
            "million" => Some(1_000_000),
            "billion" => Some(1_000_000_000),
            _ => None,
        };

        if let Some(multiplier) = scale {
            if current == 0 {
                current = 1;
            }
            total += current * multiplier;
            current = 0;
            continue;
        }

        return Err("Invalid number word".to_string());
    }

    Ok(total + current)
}

#[pyfunction(signature = (text=None))]
fn word_to_number_scalar(text: Option<String>) -> PyResult<Option<i64>> {
    let Some(value) = text else {
        return Ok(None);
    };

    match parse_word_number(&value) {
        Ok(parsed) => Ok(Some(parsed)),
        Err(message) => Err(PyValueError::new_err(message)),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_roman_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(from_roman_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(word_to_number_scalar, m)?)?;
    Ok(())
}
