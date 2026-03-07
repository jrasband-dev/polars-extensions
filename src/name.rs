use pyo3::prelude::*;

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => {
            let rest = chars.as_str().to_lowercase();
            first.to_uppercase().collect::<String>() + &rest
        }
        None => String::new(),
    }
}

fn split_on_underscore_or_space(name: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in name.chars() {
        if ch == '_' || ch.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

fn to_pascal_case_one(name: &str) -> String {
    split_on_underscore_or_space(name)
        .into_iter()
        .map(|word| capitalize_first(&word))
        .collect::<Vec<String>>()
        .join("")
}

fn to_snake_case_one(name: &str) -> String {
    let mut out = String::new();

    for (index, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && index != 0 {
            out.push('_');
        }
        out.extend(ch.to_lowercase());
    }

    out
}

fn to_camel_case_one(name: &str) -> String {
    let words = split_on_underscore_or_space(name);
    if words.is_empty() {
        return String::new();
    }

    let first = words[0].to_lowercase();
    let rest = words[1..]
        .iter()
        .map(|word| capitalize_first(word))
        .collect::<Vec<String>>()
        .join("");

    format!("{first}{rest}")
}

fn to_pascal_snake_case_one(name: &str) -> String {
    split_on_underscore_or_space(name)
        .into_iter()
        .map(|word| capitalize_first(&word))
        .collect::<Vec<String>>()
        .join("_")
}

fn to_kebab_case_one(name: &str) -> String {
    let mut out = String::new();

    for (index, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && index != 0 {
            out.push('-');
        }

        if ch == '_' {
            out.push('-');
        } else {
            out.extend(ch.to_lowercase());
        }
    }

    out
}

fn to_upper_snake_case_one(name: &str) -> String {
    let mut out = String::new();

    for (index, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && index != 0 {
            out.push('_');
        }

        if ch == '-' {
            out.push('_');
        } else {
            out.extend(ch.to_uppercase());
        }
    }

    out
}

fn to_train_case_one(name: &str) -> String {
    split_on_underscore_or_space(name)
        .into_iter()
        .map(|word| capitalize_first(&word))
        .collect::<Vec<String>>()
        .join("-")
}

#[pyfunction]
fn to_pascal_case_columns(columns: Vec<String>) -> Vec<String> {
    columns.iter().map(|name| to_pascal_case_one(name)).collect()
}

#[pyfunction]
fn to_snake_case_columns(columns: Vec<String>) -> Vec<String> {
    columns.iter().map(|name| to_snake_case_one(name)).collect()
}

#[pyfunction]
fn to_camel_case_columns(columns: Vec<String>) -> Vec<String> {
    columns.iter().map(|name| to_camel_case_one(name)).collect()
}

#[pyfunction]
fn to_pascal_snake_case_columns(columns: Vec<String>) -> Vec<String> {
    columns
        .iter()
        .map(|name| to_pascal_snake_case_one(name))
        .collect()
}

#[pyfunction]
fn to_kebab_case_columns(columns: Vec<String>) -> Vec<String> {
    columns.iter().map(|name| to_kebab_case_one(name)).collect()
}

#[pyfunction]
fn to_upper_snake_case_columns(columns: Vec<String>) -> Vec<String> {
    columns
        .iter()
        .map(|name| to_upper_snake_case_one(name))
        .collect()
}

#[pyfunction]
fn to_train_case_columns(columns: Vec<String>) -> Vec<String> {
    columns.iter().map(|name| to_train_case_one(name)).collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_pascal_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_snake_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_camel_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_pascal_snake_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_kebab_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_upper_snake_case_columns, m)?)?;
    m.add_function(wrap_pyfunction!(to_train_case_columns, m)?)?;
    Ok(())
}
