package ru.avicomp.io;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Created by @szuev on 24.10.2017.
 */
@FunctionalInterface
public interface OutputStreamSupplier {

    OutputStream open() throws IOException;

}
