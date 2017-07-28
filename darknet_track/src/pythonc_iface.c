#include "Python.h"

static PyObject *dn_callback = NULL;


PyObject *
dn_set_callback(PyObject *PyFunc)
{
    PyObject *result = NULL;

    if(!PyFunc)
        return result;

    printf("set callback..\n");

    {
        if (!PyCallable_Check(PyFunc)) {
            printf("error\n");
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        printf("valid callable found\n");
        Py_XINCREF(PyFunc);         /* Add a reference to new callback */
        Py_XDECREF(dn_callback);  /* Dispose of previous callback */
        dn_callback = PyFunc;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}


void dn_test()
{
    printf("testttttt\n");
}
