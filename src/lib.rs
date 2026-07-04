use pyo3::prelude::*;

mod name;
mod numeric;
mod io;

#[pymodule]
fn _name_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    name::register(m)?;
    numeric::register(m)?;
    io::register(m)?;
    Ok(())
}
