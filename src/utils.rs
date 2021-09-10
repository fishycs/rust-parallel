pub fn unwrap<T>(ptr: &Option<Box<T>>) -> &T {
    return &**ptr.as_ref().unwrap();
}

pub fn unwrap_mut<T>(ptr: &mut Option<Box<T>>) -> &mut T {
    return &mut **ptr.as_mut().unwrap();
}
